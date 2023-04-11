import argparse
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Generate dataset for spelling correction training (finetuning).')
    parser.add_argument('--file-path', type=str,  default='./data.json',   help='Root directory for data')
    parser.add_argument('--max-length', type=int,  default=16,   help='Maximum context length')
    parser.add_argument('--error-rate', type=float,  default=0.98,   help='Error rate of spelling mistakes in the dataset')
    parser.add_argument('--train-size', type=float,  default=0.985,   help='Fraction of dataset to be used as train dataset')
    parser.add_argument('--val-size', type=float,  default=0.01,   help='Fraction of dataset to be used as validation dataset')
    parser.add_argument('--test-size', type=float,  default=0.05,   help='Fraction of dataset to be used as test dataset')
    parser.add_argument('--output-dir', type=str,  default='./data.',   help='Output data directory')

    # Parse the arguments
    args = parser.parse_args()

    return args

def process(args):
    # Load the English data
    data_en = load_english_data(args.file_path)

    # Truncate the data according to the max_length
    data = truncate_dataset(data_en, args.max_length)

    # Clean the dataset
    cleaned_data = clean_data(data)

    # Generate pairs: (with_mistakes, correct)
    pairs = generate_dataset(cleaned_data, error_rate=args.error_rate)

    # Generate train, val, and test datasets
    train_data, test_data, val_data = split_dataset(pairs, train_size=0.985, val_size=0.01, test_size=0.05)

    # Save train_data, test_data, and val_data to files
    train_path = args.output_dir+'/train_data.json'
    val_path   = args.output_dir+'/val_data.json'
    test_path  = args.output_dir+'/test_data.json'
    save_data_to_file(train_data, train_path)
    save_data_to_file(test_data , test_path)
    save_data_to_file(val_data  , val_path)

    
if __name__ == "__main__":
    args = parse_args()
    process(args)