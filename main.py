import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from llm_api import get_llm_response

class PDF(FPDF):
    def __init__(self, skip_first_header=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_first_header = skip_first_header

    def header(self):
        self.rect(5, 5, 200, 287)
        if self.page_no() == 1 and self.skip_first_header:
            return
        self.set_font('Times', 'B', 14)
        self.cell(0, 10, 'Drug Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-20)
        self.set_font('Times', 'I', 10)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def visualize_top_drugs(results_df, output_dir):
    recommended = results_df[results_df['recommended']].nlargest(5, 'overall_score')
    avoided = results_df[~results_df['recommended']].nlargest(5, 'overall_score')

    recommended['category'] = 'Recommended'
    avoided['category'] = 'Avoided'
    top_drugs = pd.concat([recommended, avoided])

    melted = top_drugs.melt(
        id_vars=['drug_name', 'category'],
        value_vars=['compatibility_score', 'toxicity_score', 'confidence_score'],
        var_name='Metric', value_name='Score'
    )

    plt.figure(figsize=(12, 8))
    sns.barplot(data=melted, x='drug_name', y='Score', hue='Metric', palette='Set2')
    plt.title('Top 5 Recommended vs. Avoided Drugs - Scores by Metric', fontsize=16)
    plt.xlabel('Drug Name (SMILES)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_drugs_scores.png'))
    plt.close()


def heatmap_top_recommended(results_df, output_dir):
    top_recommended = results_df[results_df['recommended']].nlargest(5, 'overall_score')
    heatmap_data = top_recommended.set_index('drug_name')[['compatibility_score', 'toxicity_score', 'confidence_score']]

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'label': 'Score'})
    plt.title('Heatmap of Metrics for Top Recommended Drugs', fontsize=16)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Drug Name (SMILES)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_top_recommended.png'))
    plt.close()

def generate_drug_descriptions(json_file, output_file):
    with open(json_file, "r") as infile:
        data = json.load(infile)

    all_drugs = data['drug_analysis']['best'] + data['drug_analysis']['worst']
    enhanced_drugs = []

    for drug in all_drugs:
        prompt = f"""
        Here is a drug with the following details:
        - Drug ID: {drug['drug_id']}
        - Drug Name (SMILES): {drug['drug_name']}

        Task:
        Provide a concise, two to three-line description highlighting the drug's key features, such as unique molecular properties or structure. 
        If known, mention the disease(s) it is commonly used to treat in a seamless manner. 
        If there is no reliable information about the diseases it treats, do not include any assumptions or mention of its use case.
        The response should be a single, well-formed paragraph that is clear, crisp, and straight to the point.
        """

        try:
            response = get_llm_response(prompt)
            drug['description'] = response
        except Exception as e:
            print(f"Error processing Drug ID {drug['drug_id']}: {e}")
            drug['description'] = "Description unavailable due to an error."

        enhanced_drugs.append(drug)

    enhanced_data = {
        "drug_analysis": {
            "best": [drug for drug in enhanced_drugs if drug in data['drug_analysis']['best']],
            "worst": [drug for drug in enhanced_drugs if drug in data['drug_analysis']['worst']]
        }
    }

    with open(output_file, "w") as outfile:
        json.dump(enhanced_data, outfile, indent=4)

    print(f"Enhanced drug descriptions saved to {output_file}")


def create_pdf_with_visualizations(json_data, output_file):
    results_df = pd.DataFrame(json_data['drug_analysis']['best'] + json_data['drug_analysis']['worst'])

    output_dir = os.path.dirname(output_file)
    visualize_top_drugs(results_df, output_dir)
    heatmap_top_recommended(results_df, output_dir)

    pdf = PDF(skip_first_header=True)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font('Times', 'B', 18)
    pdf.cell(0, 10, 'Pharmaceutical Intelligence Report', 0, 1, 'C')
    pdf.set_font('Times', 'B', 14)
    pdf.cell(0, 10, 'A Fresh Look into Drug Compatibility and Toxicity', 0, 1, 'C')
    pdf.ln(10)

    pdf.set_font('Times', 'B', 14)
    pdf.cell(0, 10, 'Recommended Drugs', 0, 1)
    for drug in json_data['drug_analysis']['best']:
        pdf.set_font('Times', 'B', 12)
        pdf.cell(0, 10, f"Drug Name: {drug['drug_name']}", 0, 1)
        pdf.set_font('Times', '', 12)
        pdf.multi_cell(0, 10, f"Description: {drug['description']}")
        pdf.set_font('Times', 'B', 12)
        pdf.multi_cell(
            0,
            10,
            f"Compatibility Score: {drug['compatibility_score']:.3f}, "
            f"Toxicity Score: {drug['toxicity_score']:.3f}, "
            f"Confidence Score: {drug['confidence_score']:.3f}"
        )
        pdf.ln(5)

    pdf.add_page()
    pdf.set_font('Times', 'B', 14)
    pdf.cell(0, 10, 'Not Recommended Drugs', 0, 1)
    for drug in json_data['drug_analysis']['worst']:
        pdf.set_font('Times', 'B', 12)
        pdf.cell(0, 10, f"Drug Name: {drug['drug_name']}", 0, 1)
        pdf.set_font('Times', '', 12)
        pdf.multi_cell(0, 10, f"Description: {drug['description']}")
        pdf.multi_cell(
            0,
            10,
            f"Compatibility Score: {drug['compatibility_score']:.3f}, "
            f"Toxicity Score: {drug['toxicity_score']:.3f}, "
            f"Confidence Score: {drug['confidence_score']:.3f}"
        )
        pdf.ln(5)

    pdf.add_page()
    pdf.set_font('Times', 'B', 14)
    pdf.cell(0, 10, 'Visual Analysis', 0, 1)
    pdf.ln(7)  

    pdf.image(os.path.join(output_dir, 'top_drugs_scores.png'), x=10, y=pdf.get_y(), w=180)
    pdf.ln(120) 
    pdf.image(os.path.join(output_dir, 'heatmap_top_recommended.png'), x=12, y=pdf.get_y(), w=180)

    pdf.output(output_file)
    print(f"PDF created successfully: {output_file}")


if __name__ == "__main__":
    input_file = "output.json"
    enriched_output_file = "enriched_output.json"
    output_pdf_path = "drug_analysis_overview_visualized.pdf"

    generate_drug_descriptions(input_file, enriched_output_file)

    with open(enriched_output_file, "r") as enriched_file:
        enriched_json_data = json.load(enriched_file)

    create_pdf_with_visualizations(enriched_json_data, output_pdf_path)

    print(f"PDF report created: {output_pdf_path}")
