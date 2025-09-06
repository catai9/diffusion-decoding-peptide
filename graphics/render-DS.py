from graphviz import Digraph

# Create a directed graph
dot = Digraph(format="png", graph_attr={"rankdir": "TB"})

# Input nodes
dot.node("Input", "Input\n(sequences, precursors, memory, memory_key_padding_mask)", shape="box")
dot.node("Mass_Encoder", "Mass Encoder", shape="ellipse")
dot.node("Charge_Encoder", "Charge Encoder", shape="ellipse")
dot.node("AA_Encoder", "Amino Acid Encoder", shape="ellipse")
dot.node("Time_Encoder", "Time Encoder", shape="ellipse")

# Diffusion process
dot.node("Forward_Diffusion", "Forward Diffusion\n(Add Noise)", shape="box")
dot.node("Denoising_Network", "Denoising Network", shape="box")
dot.node("Final_Linear", "Final Linear Layer", shape="ellipse")

# Output node
dot.node("Output", "Output\n(Predicted Scores, Tokens)", shape="box")

# Connect nodes
dot.edges([
  ("Input", "Mass_Encoder"),
  ("Input", "Charge_Encoder"),
  ("Input", "AA_Encoder"),
  ("Mass_Encoder", "Forward_Diffusion"),
  ("Charge_Encoder", "Forward_Diffusion"),
  ("AA_Encoder", "Forward_Diffusion"),
  ("Forward_Diffusion", "Denoising_Network"),
  ("Time_Encoder", "Denoising_Network"),
  ("Denoising_Network", "Final_Linear"),
  ("Final_Linear", "Output")
])

# Render the graph
dot.render("decoder_architecture_DS", view=True)