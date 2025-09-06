from graphviz import Digraph

# Create a directed graph
dot = Digraph(format="png", graph_attr={"rankdir": "TB"})

# Input nodes
dot.node("Input", "Input\n(encoded_spectra, precursors, mem_masks)", shape="box")
dot.node("AA_Embedding", "Amino Acid Embedding", shape="ellipse")
dot.node("Pos_Encoding", "Positional Encoding", shape="ellipse")
dot.node("Layer_Norm", "Layer Normalization", shape="ellipse")

# Transformer Decoder
dot.node("Transformer", "Transformer Decoder\n(6 Layers)", shape="box")

# Output nodes
dot.node("Output_Proj", "Output Projection\n(Linear Layer)", shape="ellipse")
dot.node("Logits", "Logits\n[batch_size, max_length, vocab_size]", shape="box")

# Connect nodes
dot.edges([
  ("Input", "AA_Embedding"),
  ("AA_Embedding", "Pos_Encoding"),
  ("Pos_Encoding", "Layer_Norm"),
  ("Layer_Norm", "Transformer"),
  ("Transformer", "Output_Proj"),
  ("Output_Proj", "Logits")
])

# Render the graph
dot.render("decoder_architecture_DM1", view=True)