# ==========================================================
# PORTABLE HR SEMI-SYNTHETIC SIMULATION PIPELINE
# (VS Code / Local Environment Compatible)
#
# Aligned with: "Theory and Semi-synthetic simulation final"
# Total Variables: 13 SCM Nodes + candidate_id
# ==========================================================

import numpy as np
import pandas as pd
from pathlib import Path
import zipfile

# ==========================================================
# HELPER: SIGMOID FUNCTION
# ==========================================================
def sigmoid(x):
    """Maps continuous linear inputs into bounded probabilities [0,1]."""
    return 1 / (1 + np.exp(-x))


# ==========================================================
# MAIN DATA GENERATION FUNCTION (13 VARIABLES SCM + candidate_id)
# ==========================================================
def generate_hr_dataset(N, config, is_fair=False):
    """
    Generates ONE synthetic HR dataset based on the final 13-variable SCM (+ candidate_id).
    """

    # 1. ESTABLISH REPRODUCIBILITY (Same latent individuals across counterfactuals)
    np.random.seed(config['seed'])

    # ======================================================
    # PHASE 1 — ROOT VARIABLES (Independent)
    # ======================================================
    
    # Protected Demographics
    G = np.random.binomial(1, 0.5, N)              # [G] Gender (1=Female [Penalized], 0=Male)
    S = np.random.binomial(1, 0.3, N)              # [S] Immigration Status (0=Expat [Penalized], 1=Native)
    
    # Latent Merit & Time
    U_prob = np.random.normal(0, 1, N)             # [U_prob] Problem Solving Aptitude
    X_exp = np.random.poisson(5, N)                # [X_exp] Years of Experience
    
    # Systemic Confounder (Log-Normal, Independent of S)
    e_wealth = np.random.normal(0, 0.5, N)
    U_wealth = np.exp(0.5 + e_wealth)              # [U_wealth] Generational Wealth

    # ======================================================
    # PHASE 2 — DIVERSE STOCHASTIC NOISE TERMS (SUPERVISOR FIX)
    # ======================================================
    # Using non-Gaussian distributions to simulate real-world fat tails and extreme values
    e_policy = np.random.gumbel(loc=0, scale=0.5, size=N)        # Gumbel (often models extreme risk scores)
    e_behav  = np.random.laplace(loc=0, scale=0.5, size=N)       # Laplace (sharp peaks, fatter tails for human scoring)
    e_code   = np.random.standard_cauchy(N) * 0.5                # Cauchy (volatile coding test anomalies)
    e_shadow = np.random.normal(0, 0.5, N)                       # Normal
    e_uni    = np.random.normal(0, 0.5, N)                       # Normal

    # ======================================================
    # PHASE 3 — CAUSAL INTERVENTIONS (Do-Calculus)
    # ======================================================
    if not is_fair:
        # BIASED REALITY: Administrative friction is triggered for expats (S=0)
        permit_prob = sigmoid(1.0 - config['beta_permit_S'] * S)
        
        # BIASED REALITY: Human bias deducts points during interview for Women (G=1)
        gender_penalty = config['beta_behav_G'] * G * 10
        expat_penalty = config['beta_behav_S'] * (1 - S) * 10
    else:
        # COUNTERFACTUAL REALITY: Demographic connections severed
        permit_prob = sigmoid(-3.0)  # Near-zero friction for everyone
        gender_penalty = 0           # No human bias during interview
        expat_penalty = 0            # No cultural/accent bias during interview

    # ======================================================
    # PHASE 4 — OBSERVABLE ML FEATURES
    # ======================================================
    
    # Residency Permit Requirement
    X_permit = np.random.binomial(1, permit_prob)
    
    # Merit Assessments (Cleaned of demographic bias, driven by U_prob)
    raw_code = 50 + (15 * U_prob) + (2 * e_code)
    X_code = np.clip(raw_code, 0, 100) # Clipping Cauchy extremes to 0-100 rubric
    
    # Behavioral Interview Score (The Human Room)
    raw_behav = 50 + (10 * U_prob) + (2 * X_exp) - gender_penalty - expat_penalty + (5 * e_behav) 
    X_behav = np.clip(raw_behav, 0, 100)

    # Privilege Proxy (University Prestige Tiers 1=Elite, 2=Standard, 3=Unknown)
    raw_uni = config['beta_wealth'] * U_wealth + e_uni
    X_uni = np.where(raw_uni > 3.0, 1, np.where(raw_uni > 1.5, 2, 3))

    # ======================================================
    # PHASE 5 — THE HIDDEN TRAPS (Unobserved Mediators & Colliders)
    # ======================================================
    
    if not is_fair:
        # The HR Policy Trap (The Machine Room: Launders interview & permit into a penalty)
        # Higher score = Worse penalty
        H_policy = (config['beta_policy_permit'] * X_permit) + (config['beta_policy_behav'] * (100 - X_behav) / 10) + e_policy
    else:
        # Policy mechanism retained as random noise only
        H_policy = e_policy  # Pure noise, no systematic penalty 
        
    # The Shadow Network (Collider Trap requires Elite Uni AND high Cognitive Merit)
    # Inverting X_uni so Elite (1) provides a +3 boost, Standard (2) a +2, etc.
    uni_boost = 4 - X_uni 
    raw_shadow = (1.5 * uni_boost * np.maximum(U_prob, 0)) - config['relu_threshold'] + e_shadow
    C_shadow = np.maximum(0, raw_shadow)
    
    # Binary ATS Referral based on shadow network
    X_ref = np.where(C_shadow > 0, 1, 0)


    # ======================================================
    # PHASE 6 — TARGET GENERATION (Elite Job Offer)
    # ======================================================
    
    # Combining Merit, Proxies, Shadow Advantages, and HR Penalties
    target_logit = (
        0.03 * X_code + 0.02 * X_behav + 0.15 * X_exp +          # Merit
        0.5 * X_ref - 0.2 * X_uni +                              # Proxies
        config['beta_shadow_target'] * C_shadow -                # Hidden Bias (Advantage)
        H_policy                                                 # Hidden Bias (Penalty)
    )
    
    Y = np.random.binomial(1, sigmoid(target_logit))


    # ======================================================
    # PHASE 7 — DATAFRAME CONSTRUCTION (Strictly 13 Variables + candidate_id)
    # ======================================================
    
    df = pd.DataFrame({
        'candidate_id': np.arange(N),
        
        # Protected Roots
        'G_Gender': G,
        'S_ImmigStatus': S,
        
        # Unobserved Roots & Traps
        'U_wealth': U_wealth,
        'U_prob': U_prob,
        'H_policy': H_policy,
        'C_shadow': C_shadow,
        
        # Observed Features
        'X_uni': X_uni,
        'X_code': X_code,
        'X_behav': X_behav,
        'X_ref': X_ref,
        'X_exp': X_exp,
        'X_permit': X_permit,
        
        # Target
        'Y_Target': Y
    })

    return df


# ==========================================================
# FULL EXPERIMENT GRID (VS CODE BATCH GENERATOR)
# ==========================================================
def generate_dataset_grid():

    # 1. Output Directory
    output_dir = Path("HR_Simulation_Datasets")
    output_dir.mkdir(exist_ok=True)
    print("Initializing generation process...")

    # 2. Experimental Parameters (Aligned with meeting notes)
    sample_sizes = [1000, 10000]
    relu_thresholds = [1.0, 2.5]
    
    # 5 Differentiated Bias Profiles
    bias_levels = [
        {'name': 'LowBias',     'b_G': 0.5, 'b_S': 1.0, 'b_Pol_P': 0.5, 'b_Pol_B': 0.2, 'b_w': 0.2},
        {'name': 'MedBias',     'b_G': 1.0, 'b_S': 2.0, 'b_Pol_P': 1.0, 'b_Pol_B': 0.5, 'b_w': 0.5},
        {'name': 'HighBias',    'b_G': 2.0, 'b_S': 3.0, 'b_Pol_P': 2.0, 'b_Pol_B': 1.0, 'b_w': 0.8},
        {'name': 'WealthDom',   'b_G': 0.5, 'b_S': 1.0, 'b_Pol_P': 0.5, 'b_Pol_B': 0.2, 'b_w': 1.5}, # Privilege effect
        {'name': 'PolicyDom',   'b_G': 1.0, 'b_S': 1.0, 'b_Pol_P': 3.0, 'b_Pol_B': 2.0, 'b_w': 0.2}  # Aggressive AI penalty
    ]

    # 20 Seeds per configuration (Supervisor requested)
    seeds = [i * 100 for i in range(1, 21)]
    dataset_counter = 0

    # 3. Execution Loop
    for n in sample_sizes:
        for bias in bias_levels:
            for thresh in relu_thresholds:
                for seed in seeds:

                    config = {
                        'seed': seed,
                        'relu_threshold': thresh,
                        'beta_behav_G': bias['b_G'],
                        'beta_behav_S': bias['b_S'],
                        'beta_permit_S': bias['b_S'],
                        'beta_policy_permit': bias['b_Pol_P'],
                        'beta_policy_behav': bias['b_Pol_B'],
                        'beta_wealth': bias['b_w'],
                        'beta_shadow_target': 0.8
                    }

                    # Generate Base Environments
                    df_biased = generate_hr_dataset(N=n, config=config, is_fair=False)
                    df_fair = generate_hr_dataset(N=n, config=config, is_fair=True)

                    # --------------------------------------
                    # MASKING 1: FAIRNESS-AWARE DATASET
                    # Drops hidden traps & G. Keeps only observed demographic (S) visible.
                    # --------------------------------------
                    fairness_columns_to_drop = ['G_Gender', 'U_prob', 'U_wealth', 'H_policy', 'C_shadow']
                    df_biased_fairness = df_biased.drop(columns=fairness_columns_to_drop)

                    # --------------------------------------
                    # MASKING 2: UNAWARENESS DATASET
                    # Drops ALL demographics and hidden traps
                    # --------------------------------------
                    unaware_columns_to_drop = ['G_Gender', 'S_ImmigStatus', 'U_prob', 'U_wealth', 'H_policy', 'C_shadow']
                    df_biased_unaware = df_biased.drop(columns=unaware_columns_to_drop)

                    # File Naming
                    base_name = f"HR_N{n}_{bias['name']}_Thresh{thresh}_Seed{seed}"

                    # Export 4 Variations
                    df_biased.to_csv(output_dir / f"{base_name}_BIASED_ALL.csv", index=False)
                    df_biased_fairness.to_csv(output_dir / f"{base_name}_BIASED_FAIRNESS.csv", index=False)
                    df_biased_unaware.to_csv(output_dir / f"{base_name}_BIASED_MASKED.csv", index=False)
                    df_fair.to_csv(output_dir / f"{base_name}_FAIR_ALL.csv", index=False)

                    dataset_counter += 1
                    
                    # Print progress to terminal
                    if dataset_counter % 25 == 0:
                        print(f"Processed {dataset_counter} configuration grids...")

    # 4. Zip Archiving
    print("Compressing datasets into ZIP archive...")
    zip_path = Path("HR_Simulation_Datasets.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in output_dir.rglob("*.csv"):
            zipf.write(file, arcname=file.relative_to(output_dir))

    print("\nGeneration Complete.")
    print(f"Configurations generated: {dataset_counter} ({dataset_counter * 4} CSV files)")
    print(f"Saved to: {output_dir.resolve()}")
    print(f"ZIP archive created at: {zip_path.resolve()}")

# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    generate_dataset_grid()