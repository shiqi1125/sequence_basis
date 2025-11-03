import os
import argparse
import pandas as pd
import numpy as np
import pickle
import pyranges as pr
from pyfaidx import Fasta
import csv

def main():
    config = {
        'fasta': 'genomes/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna.toplevel.fa',     # 基因组FASTA文件路径
        'gtf': 'gene_models/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.52.gtf',          # 基因注释GTF文件路径
        'tpm': 'tpm_counts/zea_counts.csv',     # TPM计数文件路径
        'output': 'output.csv',      # 输出CSV文件路径
        'upstream': 1000,                    # TSS上游区域长度
        'downstream': 500,                   # TTS下游区域长度
        'padding': 20,                       # 启动子和终止子之间的填充序列长度
        'padding_char': 'N',                 # 填充字符
        'biotype': 'protein_coding',         # 要提取的基因生物类型
        'verbose': True,                     # 是否显示详细的序列信息
        'sample_size': 5                     # 显示详细信息的样本数
    }

    # 读取基因组FASTA文件
    print(f"读取基因组: {config['fasta']}")
    fasta = Fasta(config['fasta'], as_raw=True, sequence_always_upper=True, read_ahead=10000)

    # 读取TPM数据
    print(f"读取TPM数据: {config['tpm']}")
    tpm_counts = pd.read_csv(config['tpm'], index_col=0)
    
    # 根据表达量筛选高/低表达基因
    print("筛选高/低表达基因...")
    true_targets = []
    for log_count in tpm_counts['logMaxTPM'].values:
        if log_count <= np.percentile(tpm_counts['logMaxTPM'], 25):
            true_targets.append(0)  # 低表达
        elif log_count >= np.percentile(tpm_counts['logMaxTPM'], 75):
            true_targets.append(1)  # 高表达
        else:
            true_targets.append(2)  # 中等表达，不处理
    tpm_counts['true_target'] = true_targets
    
    # 只保留高/低表达基因
    tpm_counts = tpm_counts[tpm_counts['true_target'].isin([0, 1])]
    print(f"找到 {len(tpm_counts)} 个高/低表达基因")

    # 读取基因注释GTF文件
    print(f"读取基因注释: {config['gtf']}")
    gene_models = pr.read_gtf(config['gtf'], as_df=True)
    gene_models = gene_models[gene_models['Feature'] == 'gene']
    gene_models = gene_models[['Chromosome', 'Start', 'End', 'Strand', 'gene_id']]

    # 只保留TPM文件中存在的基因
    gene_models = gene_models[gene_models['gene_id'].isin(tpm_counts.index)]
    print(f"在注释文件中找到 {len(gene_models)} 个匹配的基因")

    # 创建填充序列
    padding_seq = config['padding_char'] * config['padding']

    # 准备CSV输出
    with open(config['output'], 'w', newline='') as csvfile:
        fieldnames = ['full_sequence', 'expression_level']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        processed = 0
        skipped = 0
        sample_count = 0

        for _, row in gene_models.iterrows():
            chrom = row['Chromosome']
            start = row['Start']
            end = row['End']
            strand = row['Strand']
            gene_id = row['gene_id']
            gene_name = row.get('gene_name', gene_id)  # 如果没有gene_name，则使用gene_id
            
            # 获取表达水平
            expr_level = tpm_counts.loc[gene_id, 'true_target']

            try:
                if strand == '+':
                    # 正链基因
                    # 启动子区域: TSS上游upstream到TSS下游downstream
                    prom_start = max(1, start - config['upstream'])  # 确保坐标不小于1
                    prom_end = start + config['downstream']
                    
                    # 终止子区域: TTS上游downstream到TTS下游upstream
                    term_start = end - config['downstream']
                    term_end = end + config['upstream']
                else:
                    # 负链基因
                    # 启动子区域: TTS下游upstream到TTS上游downstream (在序列上是反向的)
                    prom_start = end - config['downstream']
                    prom_end = end + config['upstream']
                    
                    # 终止子区域: TSS下游downstream到TSS上游upstream (在序列上是反向的)
                    term_start = max(1, start - config['upstream'])  # 确保坐标不小于1
                    term_end = start + config['downstream']

                # 提取序列
                promoter = fasta[chrom][prom_start-1:prom_end-1]  # pyfaidx使用0-based索引
                terminator = fasta[chrom][term_start-1:term_end-1]

                # 对于负链基因，需要取反向互补
                if strand == '-':
                    promoter = reverse_complement(promoter)
                    terminator = reverse_complement(terminator)

                # 拼接序列
                full_sequence = promoter + padding_seq + terminator
                full_sequence_length = len(full_sequence)

                # 写入CSV文件
                writer.writerow({
                    'full_sequence': full_sequence,
                    'expression_level': expr_level
                })

                processed += 1
                
                # 显示详细的序列信息
                if config['verbose'] and (sample_count < config['sample_size']):
                    print("\n" + "="*80)
                    print(f"基因: {gene_id} ({gene_name})")
                    print(f"表达水平: {'high' if expr_level == 1 else 'low'}")
                    print(f"染色体: {chrom}")
                    print(f"位置: {start}-{end} ({strand})")
                    print(f"启动子区域: {prom_start}-{prom_end} ({len(promoter)} bp)")
                    print(f"终止子区域: {term_start}-{term_end} ({len(terminator)} bp)")
                    print(f"拼接序列总长度: {full_sequence_length} bp")
                    sample_count += 1
                
                if processed % 1000 == 0:
                    print(f"已处理 {processed} 个基因")

            except Exception as e:
                print(f"跳过基因 {gene_id}: {str(e)}")
                skipped += 1

        print(f"\n处理完成!")
        print(f"成功处理: {processed} 个基因")
        print(f"跳过: {skipped} 个基因")
        print(f"结果已保存到: {config['output']}")
        
        if config['verbose']:
            print(f"\n已显示 {sample_count} 个基因的详细信息")

def reverse_complement(seq):
    """计算DNA序列的反向互补序列"""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 
                  'N': 'N', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a', 'n': 'n'}
    return ''.join([complement.get(base, base) for base in reversed(seq)])

if __name__ == "__main__":
    main()
