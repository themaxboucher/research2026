from collect import collect_dataset
from analyse import analyse_dataset
from generate import generate_comments_for_dataset
import argparse
import logging

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    args = parse_args()
    if args.collect:
        collect_dataset()
    if args.generate:
        generate_comments_for_dataset()
    if args.analyse:
        analyse_dataset()

if __name__ == "__main__":
    main()
