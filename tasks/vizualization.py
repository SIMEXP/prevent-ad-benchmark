from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

sns.set_theme(style="whitegrid")

def toSnakeCase(string):
    string = re.sub(r'(?<=[a-z])(?=[A-Z])|[^a-zA-Z]', ' ', string).strip().replace(' ', '_')
    return ''.join(string.lower())



def plot_demographics(demo, title='Demographic Distribution at Baseline'):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.set_size_inches(12, 10)
    fig.suptitle(title, fontsize=16)
    sns.histplot(data=demo, x='Candidate_Age', hue='Sex', bins=30, kde=True, stat='density', common_norm=False, ax=ax1)
    ax1.set_title('Age Distribution by Sex at Baseline')
    ax1.set_xlabel('Candidate Age (years)')
    ax1.set_ylabel('Density')
    ax1.set_ylim(0, 0.15)
    ax1.legend(title='Sex', labels=['Female', 'Male'])  

    sns.countplot(data=demo, x='Sex', ax=ax2, order=['Female', 'Male'])
    ax2.set_title('Sex Distribution at Baseline')
    ax2.set_xlabel('Sex')
    ax2.set_ylabel('Count')
    ax2.set_ylim(0, 300)
    for p in ax2.patches:
        ax2.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),  
                    ha='center', va='center', xytext = (0, -15), 
                    textcoords = 'offset points',
                    color='white')

    sns.histplot(data=demo, x='cerebellum_coverage', hue='Sex', bins=30, kde=True, stat='density', common_norm=False, ax=ax3)
    ax3.set_title('Cerebellum Coverage Distribution by Sex at Baseline')
    ax3.set_xlabel('Cerebellum Coverage (proportion)')
    ax3.set_ylabel('Density')
    ax3.set_ylim(0, 12)
    if demo['cerebellum_coverage'].min() < 0.5:
        ax3.axvline(x=0.75, color='red', linestyle='--', label='QC Threshold')
        ax3.legend(title='Sex', labels=['Female', 'Male', 'QC Threshold'])  
    else:
        ax3.legend(title='Sex', labels=['Female', 'Male'])  

    sns.histplot(data=demo, x='proportion_kept', hue='Sex', bins=30, kde=True, stat='density', common_norm=False, ax=ax4)
    ax4.set_title('Proportion of Low Motion Volumes by Sex at Baseline')
    ax4.set_xlabel('Proportion of Low Motion Volumes (proportion)')
    ax4.set_ylabel('Density')
    ax4.set_ylim(0, 35)
    if demo['proportion_kept'].min() < 0.5:
        ax4.axvline(x=0.5, color='red', linestyle='--', label='QC Threshold')
        ax4.legend(title='Sex', labels=['Female', 'Male', 'QC Threshold'])  
    else:
        ax4.legend(title='Sex', labels=['Female', 'Male'])  

    plt.tight_layout()
    plt.savefig(f'outputs/{toSnakeCase(title)}.png', dpi=300)
    plt.close() 

qc_pheno = "data/source/dataset-preventad_version-8.1internal_pipeline-gigapreprocess2/dataset-preventad81internal_desc-sexage_pheno.tsv"
demo = pd.read_csv(qc_pheno, sep='\t')
demo['Candidate_Age'] = demo['Candidate_Age'] / 12 # convert candidate age to year
demo = demo.loc[demo['ses'] == "BL00", :]
demo = demo.loc[demo['run'] == 1, :]
plot_demographics(demo, title='Demographic Distribution before QC Filtering')
demo = demo.loc[demo['cerebellum_coverage']>0.75, :]
demo = demo.loc[demo['proportion_kept']>0.5, :]
plot_demographics(demo, title='Demographic Distribution after QC Filtering')
