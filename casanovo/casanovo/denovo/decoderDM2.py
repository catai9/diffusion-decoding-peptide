import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Tuple, Dict, Union
from depthcharge.masses import PeptideMass


class DiffusionDecoderDM2(nn.Module):
    """
    A diffusion-based decoder for peptide sequence generation.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality used by the transformer model.
    max_length : int
        The maximum peptide length to decode.
    residues : Union[Dict[str, float], str]
        The amino acid dictionary and their masses. Can be "canonical", "massivekb",
        or a custom dictionary.
    max_charge : int
        The maximum precursor charge to consider.
    n_steps : int
        Number of diffusion steps.
    dropout : float
        Dropout probability for transformer layers.
    """

    def __init__(
        self,
        dim_model: int = 512,
        max_length: int = 100,
        residues: Union[Dict[str, float], str] = "canonical",
        max_charge: int = 5,
        n_steps: int = 100,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.max_length = max_length
        self.max_charge = max_charge
        self.n_steps = n_steps
        self.dropout = dropout

        # Initialize residues
        self.residues = self._initialize_residues(residues)
        self.peptide_mass_calculator = PeptideMass(self.residues)
        self._aa2idx = {aa: i + 1 for i, aa in enumerate(sorted(self.residues.keys()))}
        self._aa2idx["$"] = 0  # Padding/stop token
        self._idx2aa = {i: aa for aa, i in self._aa2idx.items()}
        self.vocab_size = len(self._aa2idx)

        # Positional encoding
        self.pos_encoder = nn.Parameter(
            self._get_positional_encoding(max_length, dim_model), requires_grad=False
        )

        self.time_embedding = nn.Embedding(self.n_steps, self.dim_model)

        # Transformer decoder layer
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=8,
            dim_feedforward=dim_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=6
        )

        # Output projection
        self.output_proj = nn.Linear(dim_model, self.vocab_size)

        # Diffusion parameters
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(self.device)

        # Embedding for amino acids
        self.aa_embedding = nn.Embedding(self.vocab_size, dim_model)

        # Layer normalization
        self.norm = nn.LayerNorm(dim_model)

    def _initialize_residues(self, residues: Union[Dict[str, float], str]) -> Dict[str, float]:
        """
        Initialize the residue dictionary based on input specification.
        """
        if residues == "canonical":
            return {
                "G": 57.021463735,
                "A": 71.037113805,
                "S": 87.032028435,
                "P": 97.052763875,
                "V": 99.068413945,
                "T": 101.047678505,
                "C+57.021": 103.009184505 + 57.02146,
                "L": 113.084064015,
                "I": 113.084064015,
                "N": 114.042927470,
                "D": 115.026943065,
                "Q": 128.058577540,
                "K": 128.094963050,
                "E": 129.042593135,
                "M": 131.040484645,
                "H": 137.058911875,
                "F": 147.068413945,
                # "U": 150.953633405,
                "R": 156.101111050,
                "Y": 163.063328575,
                "W": 186.079312980,
                # "O": 237.147726925,
            }
        elif residues == "massivekb":
            # Placeholder for MassIVE-KB residues; extend as needed
            base = self._initialize_residues("canonical")
            base.update({
                # N-terminal mods:
                "+42.011": 42.010565,  # Acetylation
                "+43.006": 43.005814,  # Carbamylation
                "-17.027": -17.026549,  # NH3 loss
                "+43.006-17.027": (43.006814 - 17.026549),
                # AA mods:
                "M+15.995": 131.040484645 + 15.994915,  # Met Oxidation
                "N+0.984": 114.042927470 + 0.984016,  # Asn Deamidation
                "Q+0.984": 128.058577540 + 0.984016,  # Gln Deamidation
            })  # Example modification
            return base
        elif isinstance(residues, dict):
            return residues
        else:
            raise ValueError("residues must be 'canonical', 'massivekb', or a dictionary")

    def _get_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Generate positional encodings.
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    @property
    def device(self) -> torch.device:
        """
        Get the device of the model parameters.
        """
        return next(self.parameters()).device

    def forward(
        self,
        encoded_spectra: torch.Tensor,
        precursors: torch.Tensor,
        mem_masks: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the diffusion decoder using denoising steps on continuous embeddings.
        """
        batch_size = encoded_spectra.size(0)
        device = encoded_spectra.device

        if t is None:
            t = torch.full((batch_size,), self.n_steps - 1, dtype=torch.long, device=device)

        # Ground truth embeddings as the "clean" target (x0)
        x_0 = torch.randint(1, self.vocab_size, (batch_size, self.max_length), device=device)
        x_0_emb = self.aa_embedding(x_0).to(device)  # (B, L, D)

        # Forward diffuse (q(x_t | x_0))
        alpha_bar_t = self.alpha_bar[t.cpu()].view(batch_size, 1, 1).to(device)  # (B, 1, 1)
        noise = torch.randn_like(x_0_emb, device=device)
        x_t = torch.sqrt(alpha_bar_t) * x_0_emb + torch.sqrt(1 - alpha_bar_t) * noise

        # Optional: Add time embeddings
        # (e.g., sinusoidal or learned embedding from t, shape: [B, D])
        t_emb = self.time_embedding(t)  # (B, D)
        t_emb = t_emb.unsqueeze(1).expand(-1, self.max_length, -1)
        x_t = x_t + t_emb

        # Positional encoding
        pos_enc = self.pos_encoder[:self.max_length].unsqueeze(0).to(device)  # (1, L, D)
        x_t = x_t + pos_enc
        x_t = self.norm(x_t)

        # Generate target mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.max_length).to(device)

        # Transformer decoding
        output = self.transformer_decoder(
            tgt=x_t,
            memory=encoded_spectra,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=mem_masks,
        )

        logits = self.output_proj(output)
        return logits


    def decode(
        self,
        encoded_spectra: torch.Tensor,
        precursors: torch.Tensor,
        mem_masks: torch.Tensor,
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Decode peptide sequences using a diffusion-based approach.

        Parameters
        ----------
        encoded_spectra : torch.Tensor
            Encoded spectra, shape (batch_size, seq_len, dim_model).
        precursors : torch.Tensor
            Precursor information, shape (batch_size, 3).
        mem_masks : torch.Tensor
            Memory masks, shape (batch_size, seq_len).

        Returns
        -------
        List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list of top peptide predictions with score,
            amino acid scores, and sequence.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(encoded_spectra, precursors, mem_masks)
            probs = torch.softmax(logits, dim=-1)
            seq_indices = logits.argmax(dim=-1)

        batch_size = encoded_spectra.size(0)
        results = []
        for i in range(batch_size):
            seq = seq_indices[i].cpu().numpy()
            prob = probs[i].cpu().numpy()
            peptide = []
            aa_scores = []
            for idx, p in zip(seq, prob):
                if idx == 0:  # Stop token
                    break
                aa = self._idx2aa.get(idx, "$")
                if aa == "$":
                    break
                peptide.append(aa)
                aa_scores.append(p[idx])
            peptide_str = "".join(peptide)
            score = float(np.prod(aa_scores)) if aa_scores else 0.0
            aa_scores = np.array(aa_scores)
            results.append([(score, aa_scores, peptide_str)])

        return results

    def training_step(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        peptides: List[str],
        encoder: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a training step for the decoder.

        Parameters
        ----------
        spectra : torch.Tensor
            MS/MS spectra, shape (batch_size, n_peaks, 2).
        precursors : torch.Tensor
            Precursor information, shape (batch_size, 3).
        peptides : List[str]
            Ground truth peptide sequences.
        encoder : nn.Module
            The spectrum encoder.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Predicted logits and ground truth indices.
        """
        # Encode spectra
        encoded_spectra, mem_masks = encoder(spectra)

        # Convert peptides to indices
        device = encoded_spectra.device
        batch_size = spectra.size(0)
        truth = torch.zeros(batch_size, self.max_length, dtype=torch.long, device=device)
        for i, pep in enumerate(peptides):
            for j, aa in enumerate(pep[:self.max_length]):
                truth[i, j] = self._aa2idx.get(aa, 0)

        # Random diffusion timestep
        t = torch.randint(1, self.n_steps, (batch_size,), device=device)

        # Forward pass
        logits = self.forward(encoded_spectra, precursors, mem_masks, t)

        return logits, truth
        