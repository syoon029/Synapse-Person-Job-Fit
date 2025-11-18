import pandas as pd
import matplotlib.pyplot as plt
import io

# Load your data
df = pd.read_csv("evolution_history.csv")
# 1. Establish Baseline (Gen 0) for each Resume
baseline = df[df['Generation'] == 0][['Resume_Index', 'Best_Fitness']]
baseline = baseline.rename(columns={'Best_Fitness': 'Base_Fitness'})
df_merged = df.merge(baseline, on='Resume_Index')

# 2. Calculate Percentage Improvement
df_merged['Pct_Improvement'] = (
    (df_merged['Best_Fitness'] - df_merged['Base_Fitness']) / 
    df_merged['Base_Fitness']
) * 100

# 3. Aggregate Stats per Generation
agg_stats = df_merged.groupby('Generation')['Pct_Improvement'].agg(['mean', 'std']).reset_index()

# 4. Plotting
plt.figure(figsize=(10, 6))

# Plot individual trajectories (Spaghetti Plot)
# CHANGED: used 'gray' instead of 'lightgray' and increased alpha to 0.6
for resume_idx in df_merged['Resume_Index'].unique():
    subset = df_merged[df_merged['Resume_Index'] == resume_idx]
    plt.plot(subset['Generation'], subset['Pct_Improvement'], 
             color='gray', alpha=0.6, linewidth=1)

# Plot Aggregate Mean
plt.plot(agg_stats['Generation'], agg_stats['mean'], 
         color='#007acc', linewidth=3, marker='o', label='Average Improvement')

# Add Shaded Region (Mean +/- Std Dev)
plt.fill_between(agg_stats['Generation'], 
                 agg_stats['mean'] - agg_stats['std'], 
                 agg_stats['mean'] + agg_stats['std'], 
                 color='#007acc', alpha=0.15, label='Standard Deviation')

plt.title('Relative Fitness Improvement Over Baseline Resume')
plt.xlabel('Iteration')
plt.ylabel('Ensemble Fitness Score (Phase II) Improvement (%)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('resume_improvement_darker.png')