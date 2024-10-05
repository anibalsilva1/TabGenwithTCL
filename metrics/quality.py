import pandas as pd


from sdmetrics.reports.single_table import QualityReport, DiagnosticReport

from .utils import meta_for_sdmetrics

def evaluate_quality_report(
        df_real: pd.DataFrame, 
        df_syn: pd.DataFrame, 
        meta: dict
    ) -> dict:
    '''
    Computes similarity and correlation between real and synthetic columns.
    The marginal is computed for each column independently, between real and synthetic data.
        - If a column is of type numerical, then the Kolmogorov-Smirnov is computed. https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/kscomplement
        - If a column is of type categorical, then the Total Variation Distance is computed. https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/tvcomplement

    In the end, all the scores are averaged.
    
    The pairs correlation computes the correlation between two columns, for both synthetic and real data, and then compares the correlations obtained by a score function:
    https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/correlationsimilarity#how-does-it-work
    
    From: https://github.com/sdv-dev/SDMetrics/blob/main/sdmetrics/reports/single_table/_properties/column_pair_trends.py
        If one is comparing a continuous column to a discrete column, use the discrete version
        of the continuous column and the Contingency metric. Otherwise use the original columns.
        If the columns are both continuous, use the Correlation metric. If the columns are both
        discrete, use the Contingency metric. 

    In the end, all the scores are averaged.
    '''
    df_real = df_real.astype(df_syn.dtypes) # noticed that some types were unmatched.

    sdmetrics_meta = meta_for_sdmetrics(meta)
    quality_report = QualityReport()
    quality_report.generate(df_real, df_syn, sdmetrics_meta, verbose=False)
    quality = quality_report.get_properties()

    marginal = quality['Score'][0]
    pairs_correlation = quality['Score'][1]
                                        
    diagnostic_report = DiagnosticReport()
    diagnostic_report.generate(df_real, df_syn, sdmetrics_meta, verbose=False)

    # Both must return 1.0, else something is wrong with the format of dataframes.
    diagnostic = diagnostic_report.get_properties()
    data_validity = diagnostic['Score'][0]
    data_struct = diagnostic['Score'][1]

    report = {
        'marginal': marginal,
        'pairs_correlation': pairs_correlation,
        'data_validity': data_validity,
        'data_struct': data_struct
    }
    return report