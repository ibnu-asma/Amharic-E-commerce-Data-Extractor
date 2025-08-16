#!/usr/bin/env python3
"""
Task 2: CoNLL Format Labeling
Automated script to generate CoNLL format labeled data for NER training
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_task2():
    """Execute Task 2: CoNLL Format Labeling"""
    print("🚀 STARTING TASK 2: CoNLL FORMAT LABELING")
    print("=" * 50)
    
    # Step 1: Ensure data is processed
    print("\n📊 Step 1: Processing data for CoNLL format...")
    try:
        from src.preprocessing.process_data import process_all_raw_files
        process_all_raw_files()
        print("✅ Data processing complete")
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return False
    
    # Step 2: Generate CoNLL labeled data
    print("\n🏷️  Step 2: Generating CoNLL labeled data...")
    try:
        from src.labeling.label_conll import main as label_main
        label_main()
        print("✅ CoNLL labeling complete")
    except Exception as e:
        print(f"❌ CoNLL labeling failed: {e}")
        return False
    
    # Step 3: Validate output
    print("\n✅ Step 3: Validating CoNLL output...")
    output_file = 'data/labeled/conll_labeled.txt'
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        token_lines = [l for l in lines if l.strip() and '\t' in l]
        sentences = len([l for l in lines if not l.strip()])
        
        print(f"📈 CoNLL Statistics:")
        print(f"   - Total tokens: {len(token_lines)}")
        print(f"   - Total sentences: {sentences}")
        print(f"   - Output file: {output_file}")
        
        # Show sample
        print(f"\n📝 Sample CoNLL format:")
        for line in lines[:10]:
            if line.strip():
                print(f"   {line.strip()}")
        
        print("\n🎉 TASK 2 COMPLETED SUCCESSFULLY!")
        return True
    else:
        print("❌ CoNLL output file not found")
        return False

if __name__ == "__main__":
    success = run_task2()
    if success:
        print("\n🎯 Ready for Task 3: Model Training!")
    else:
        print("\n⚠️  Task 2 incomplete. Check errors above.")