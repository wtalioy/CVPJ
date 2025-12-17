import pandas as pd
import argparse
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def evaluate_csv_accuracy(ground_truth_file, prediction_file, output_file=None):
    """
    比较两个CSV文件的准确度
    
    参数:
        ground_truth_file: 标准答案文件路径
        prediction_file: 预测结果文件路径
        output_file: 输出详细结果的CSV文件路径（可选）
    """
    
    try:
        # 读取标准答案文件
        print(f"正在读取标准答案文件: {ground_truth_file}")
        ground_truth_df = pd.read_csv(ground_truth_file)
        
        # 读取预测结果文件
        print(f"正在读取预测结果文件: {prediction_file}")
        prediction_df = pd.read_csv(prediction_file)
        
        # 检查文件格式
        if 'image_id' not in ground_truth_df.columns or 'label' not in ground_truth_df.columns:
            raise ValueError("标准答案文件必须包含 'image_id' 和 'label' 列")
            
        if 'image_id' not in prediction_df.columns or 'label' not in prediction_df.columns:
            raise ValueError("预测结果文件必须包含 'image_id' 和 'label' 列")
        
        # 检查数据行数
        if len(ground_truth_df) != 1000:
            print(f"警告: 标准答案文件有 {len(ground_truth_df)} 行，期望 1000 行")
        
        if len(prediction_df) != 1000:
            print(f"警告: 预测结果文件有 {len(prediction_df)} 行，期望 1000 行")
        
        # 跳过表头，获取有效数据
        ground_truth_labels = ground_truth_df['label']
        prediction_labels = prediction_df['label']
        
        # 检查标签一致性
        if len(ground_truth_labels) != len(prediction_labels):
            print(f"警告: 有效数据行数不一致 - 标准答案: {len(ground_truth_labels)}, 预测结果: {len(prediction_labels)}")
            # 取最小长度进行比较
            min_length = min(len(ground_truth_labels), len(prediction_labels))
            ground_truth_labels = ground_truth_labels[:min_length]
            prediction_labels = prediction_labels[:min_length]
        
        # 计算各项指标
        accuracy = accuracy_score(ground_truth_labels, prediction_labels)
        precision = precision_score(ground_truth_labels, prediction_labels, average='binary')
        recall = recall_score(ground_truth_labels, prediction_labels, average='binary')
        f1 = f1_score(ground_truth_labels, prediction_labels, average='binary')
        
        # 计算混淆矩阵
        cm = confusion_matrix(ground_truth_labels, prediction_labels)
        tn, fp, fn, tp = cm.ravel()
        
        # 输出结果
        print("\n" + "="*50)
        print("CSV文件准确度评估结果")
        print("="*50)
        print(f"标准答案文件: {ground_truth_file}")
        print(f"预测结果文件: {prediction_file}")
        print(f"比较样本数: {len(ground_truth_labels)}")
        print("\n评估指标:")
        print(f"准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        print("\n混淆矩阵:")
        print(f"真阴性 (TN): {tn}")
        print(f"假阳性 (FP): {fp}")
        print(f"假阴性 (FN): {fn}")
        print(f"真阳性 (TP): {tp}")
        
        # 计算错误分类的样本
        incorrect_indices = np.where(ground_truth_labels != prediction_labels)[0]
        correct_indices = np.where(ground_truth_labels == prediction_labels)[0]
        
        print(f"\n错误分类样本数: {len(incorrect_indices)}")
        print(f"正确分类样本数: {len(correct_indices)}")
        
        # 如果指定了输出文件，保存详细结果
        if output_file:
            # 创建详细结果DataFrame
            result_df = ground_truth_df.iloc[1:].copy()
            result_df = result_df.iloc[:len(ground_truth_labels)].copy()
            result_df['predicted_label'] = prediction_labels
            result_df['is_correct'] = (ground_truth_labels == prediction_labels)
            
            # 保存结果
            result_df.to_csv(output_file, index=False)
            print(f"\n详细结果已保存到: {output_file}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'incorrect_count': len(incorrect_indices),
            'correct_count': len(correct_indices)
        }
        
    except FileNotFoundError as e:
        print(f"错误: 文件不存在 - {e}")
        return None
    except Exception as e:
        print(f"错误: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='比较两个CSV文件的准确度')
    parser.add_argument('--ground_truth', default='dataset\\测试集标签\\ai图像检测.csv', 
                       help='标准答案文件路径 (默认: dataset\\测试集标签\\ai图像检测.csv)')
    parser.add_argument('--prediction', default='results/result.csv', 
                       help='预测结果文件路径 (默认: results/result.csv)')
    parser.add_argument('--output', help='输出详细结果的CSV文件路径 (可选)')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.ground_truth):
        print(f"错误: 标准答案文件不存在 - {args.ground_truth}")
        return
    
    if not os.path.exists(args.prediction):
        print(f"错误: 预测结果文件不存在 - {args.prediction}")
        print("请先创建预测结果文件或指定正确的文件路径")
        return
    
    # 执行评估
    evaluate_csv_accuracy(args.ground_truth, args.prediction, args.output)

if __name__ == "__main__":
    main()
