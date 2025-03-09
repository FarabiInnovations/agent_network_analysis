import pandas as pd
import matplotlib.pyplot as plt

# ------------
# Creating the Agent Registry table. 
# Agent registry (AR): A detailed source documenting each agent’s role within the system, aligned with
#  an Agent RACI matrix (Responsible, Accountable, Consulted, Informed). This includes an agent 
#  unique identifier (UID), job description, persona and policy UIDs. The agent registry also 
#  identifies the primary Foundational Model UID, source system (vendor or internal), model 
#  dependency UIDs, ACP UID, fallback mechanism and model bench UIDs, and type-I and type-II 
#  defined error thresholds (scaling factors).
# https://medium.com/@ryanfattini/franchise-furanchaizu-automata-enterprise-alignment-with-emergent-potential-92545cabe462
# ------------

agent_registry_data = {
    "Agent UID": ["A", "B", "C", "D", "E"],
    "Role": [
        "User-Facing Assistant",
        "User-Facing Assistant",
        "Evaluation & Refinement",
        "Evaluation & Refinement",
        "Evaluation & Refinement"
    ],
    "RACI": [
        "Responsible, Accountable",
        "Responsible, Accountable",
        "Consulted, Informed",
        "Consulted, Informed",
        "Consulted, Informed"
    ],
    "Job Description ID": ["JD_A", "JD_B", "JD_C", "JD_D", "JD_E"],
    "Job Summary": [
        "Handles direct user interactions, provides responses, and executes commands.",
        "Handles direct user interactions, provides responses, and executes commands.",
        "Evaluates and refines output from A and B to ensure quality.",
        "Performs additional checks and optimizations on refined outputs.",
        "Final validation before responses are confirmed."
    ],
    "Persona UID": ["P_A1", "P_B1", "P_C1", "P_D1", "P_E1"],
    "Policy UID": ["Pol_A1", "Pol_B1", "Pol_C1", "Pol_D1", "Pol_E1"], # system and boundary rules with function and specialization
    "Primary FM UID": ["Claude Sonnet 3.7", "FM_002", "FM_003", "FM_004", "FM_005"],
    "Source System": ["IBM_WatsonX", "AWS_Sagemaker", "Internal", "Internal", "Internal"],
    "Model Service UID": ["Human_Resources:Recruiting:Sales:Sreening", "Dept:SubDept:Team:Function", "Dept_C1", "Dept_D1", "Dept_E1"],
    "Model SME UID": ["Director HR", "VP SC", "VP Data", "VP Data", "VP Data"],
    "ACP UID": ["ACP_A1", "ACP_B1", "ACP_C1", "ACP_D1", "ACP_E1"],
    "Fallback Mechanism": [
        "Rule-based system fallback",
        "Rule-based system fallback",
        "Human oversight",
        "Statistical validation",
        "Majority vote mechanism"
    ],
    "Model Bench UID": [
        ["MB_A1", "MB_A2"],  # Two fallback models for Agent A
        ["MB_B1"],           # Single model for Agent B
        ["MB_C1", "MB_C2", "MB_C3"],  # Three fallbacks models for C
        ["MB_D1", "MB_D2"],  # Two fallbacks Agent D
        ["MB_E1"]            # Single model for Agent E
    ],
    "Type-I Error Threshold": [0.05, 0.05, 0.02, 0.02, 0.01], # defined by SME UID (subject matter expert)
    "Type-II Error Threshold": [0.1, 0.1, 0.05, 0.03, 0.02],  # defined by SME UID (subject matter expert)
    "Error_Priority": [1,1,1,2,2] # defined by SME UID (subject matter expert)
}

# convert to DataFrame
agent_registry_df = pd.DataFrame(agent_registry_data)

#print(agent_registry_df.to_string())

# ------------------------------------#
# Table
#-------------------------------------#
fig, ax = plt.subplots(figsize=(12, 6))
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)
table = ax.table(cellText=agent_registry_df.values,
                 colLabels=agent_registry_df.columns,
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.auto_set_column_width([i for i in range(len(agent_registry_df.columns))]) 
plt.title("Agent Registry Table", fontsize=12)
plt.show()

