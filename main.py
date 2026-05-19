from collect import collect_dataset
from analyse import analyse_dataset
from generate import generate_comments_for_dataset
import argparse

# Knowledge cutoff dates
# Gemini 3.1 Pro (https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-pro): January 2025
# GPT-5.5 (https://developers.openai.com/api/docs/models/gpt-5.5): December 1st, 2025
# Llama-3.1-8b (https://huggingface.co/meta-llama/Llama-3.1-8B): December 2023
# Qwen...


CUTOFF_DATE = "2025-02-01"
REPO_LIMIT = 20
COMMIT_LIMIT = 500

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true", help="Collect data from GitHub")                                
    parser.add_argument("--generate", action="store_true", help="Generate comments for collected data")
    parser.add_argument("--analyse", action="store_true", help="Analyse collected data") 
    args = parser.parse_args()

    # If neither was passed, treat it as "run both"               
    if not args.collect and not args.generate and not args.analyse:                     
        args.collect = True                                       
        args.generate = True
        args.analyse = True                                       
    return args 

def main():
    args = parse_args()
    if args.collect :
        collect_dataset(CUTOFF_DATE, REPO_LIMIT, COMMIT_LIMIT)
    if args.analyse:
        analyse_dataset()
    if args.generate:
        generate_comments_for_dataset()

if __name__ == "__main__":
    main()
