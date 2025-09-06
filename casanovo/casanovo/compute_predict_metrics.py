import depthcharge
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import argparse

from sklearn.metrics import auc
from casanovo.denovo import evaluate


parser = argparse.ArgumentParser(description='Evaluate the predicted output from casanovo model')
parser.add_argument('--actual_file', help='annotated test file', type=str, default='/home/amytai/casanovo/sample_data/sample_preprocessed_spectra.mgf')
parser.add_argument('--predicted_file', help='predicted file', type=str, default="/home/amytai/casanovo/casanovo_20250409112342.mztab")
parser.add_argument('--output_file', help='csv file name', type=str, default='sample_test.csv')
parser.add_argument('--plot_file', help='plot file name', type=str, default='sample_test_prec_cov.png')
parser.add_argument('--knapsack_output', help='If knapsack was used', action='store_true')

args = parser.parse_args()

print(f"Arguments Used: ")
print(f"actual_file: {args.actual_file}")
print(f"predicted_file: {args.predicted_file}")
print(f"output_file: {args.output_file}")
print(f"plot_file: {args.plot_file}")
print(f"knapsack_output: {args.knapsack_output}")


def extract_values_after_SEQ(file_path):
    rows = []

    with open(file_path, 'r') as file:
        for line in file:
            if 'SEQ=' in line:
                values = line.split('SEQ=')[1].strip()
                rows.append([values])

    df = pd.DataFrame(rows, columns=['SEQ_Values'])

    return df

def read_mztab(file_path):
    header = [
        "PSH", "sequence", "PSM_ID", "accession", "unique", "database", 
        "database_version", "search_engine", "search_engine_score[1]", 
        "modifications", "retention_time", "charge", "exp_mass_to_charge", 
        "calc_mass_to_charge", "spectra_ref", "pre", "post", "start", "end", 
        "opt_ms_run[1]_aa_scores"
    ]
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("\t".join(header)):
                break

        data = []
        for line in file:
            data.append(line.strip().split('\t'))

    df = pd.DataFrame(data, columns=header)
    return df


def append_csv_columnwise(df1, df2, output_file):
    if len(df1) != len(df2):
        print("Warning: The number of rows in both files do not match.")
    
    df_appended = pd.concat([df1, df2], axis=1)
    
    df_appended.to_csv(output_file, index=False)
    print(f"Files have been appended column-wise and saved to {output_file}")

    return df_appended


df_actual = extract_values_after_SEQ(args.actual_file)
df_pred = read_mztab(args.predicted_file)

if args.knapsack_output:
    def remove_dollar(str_lst):
        lst = ast.literal_eval(str_lst)
        return ''.join([char for char in lst if char != '$'])

    df_pred['cleaned_sequences'] = df_pred['sequence'].apply(remove_dollar)
else:
    df_pred['cleaned_sequences'] = df_pred['sequence']

df_appended = append_csv_columnwise(df_actual, df_pred, args.output_file)

psm_sequences = df_appended

print(psm_sequences.head(2))

psm_sequences["search_engine_score[1]"] = psm_sequences["search_engine_score[1]"].astype(float)


# Copied from https://casanovo.readthedocs.io/en/latest/faq.html
# Sort the PSMs by descreasing prediction score.
psm_sequences = psm_sequences.sort_values(
    "search_engine_score[1]", ascending=False
)

# Find matches between the true and predicted peptide sequences.
aa_matches_batch, n_aa1, n_aa2 = evaluate.aa_match_batch(
    psm_sequences["SEQ_Values"],
    psm_sequences["cleaned_sequences"],
    depthcharge.masses.PeptideMass("massivekb").masses,
)

# Calculate the peptide precision and coverage.
peptide_matches = np.asarray([aa_match[1] for aa_match in aa_matches_batch])
precision = np.cumsum(peptide_matches) / np.arange(1, len(peptide_matches) + 1)
coverage = np.arange(1, len(peptide_matches) + 1) / len(peptide_matches)

# Calculate the score threshold at which peptide predictions don't fit the
# precursor m/z tolerance anymore.
threshold = np.argmax(psm_sequences["search_engine_score[1]"] < 0)

# Print the performance values.
print(f"Peptide precision = {precision[threshold]:.3f}")
print(f"Coverage = {coverage[threshold]:.3f}")
print(f"Peptide precision @ coverage=1 = {precision[-1]:.3f}")

# Plot the precision–coverage curve.
width = 4
height = width / 1.618
fig, ax = plt.subplots(figsize=(width, width))

ax.plot(
    coverage, precision, label=f"Casanovo AUC = {auc(coverage, precision):.3f}"
)
ax.scatter(
    coverage[threshold],
    precision[threshold],
    s=50,
    marker="D",
    edgecolors="black",
    zorder=10,
)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.set_xlabel("Coverage")
ax.set_ylabel("Peptide precision")
ax.legend(loc="lower left")

plt.savefig(args.plot_file, dpi=300, bbox_inches="tight")
plt.close()


# NEW: Get amino acid metrics
aa_precision, aa_recall, pep_precision = evaluate.aa_match_metrics(aa_matches_batch, n_aa1, n_aa2)
print(f"aa_precision = {aa_precision:.3f}")
print(f"aa_recall = {aa_recall:.3f}")
print(f"pep_precision = {pep_precision:.3f}")


