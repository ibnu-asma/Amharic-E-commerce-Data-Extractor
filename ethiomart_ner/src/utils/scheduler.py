# import schedule
# import time
# import subprocess
# import logging
# from datetime import datetime

# def run_scraper():
#     """Run the scraper script."""
#     try:
#         result = subprocess.run(['python', 'src/data_ingestion/scraper.py'], 
#                               capture_output=True, text=True)
#         if result.returncode == 0:
#             print(f"[{datetime.now()}] Scraper completed successfully")
#         else:
#             print(f"[{datetime.now()}] Scraper failed: {result.stderr}")
#     except Exception as e:
#         print(f"[{datetime.now()}] Error running scraper: {e}")

# def run_processor():
#     """Run the data processor."""
#     try:
#         result = subprocess.run(['python', 'src/preprocessing/process_data.py', '--all'], 
#                               capture_output=True, text=True)
#         if result.returncode == 0:
#             print(f"[{datetime.now()}] Processor completed successfully")
#         else:
#             print(f"[{datetime.now()}] Processor failed: {result.stderr}")
#     except Exception as e:
#         print(f"[{datetime.now()}] Error running processor: {e}")

# def setup_scheduler():
#     """Setup automated data collection schedule."""
#     # Schedule scraping every 6 hours
#     schedule.every(6).hours.do(run_scraper)
    
#     # Schedule processing 30 minutes after scraping
#     schedule.every(6).hours.at(":30").do(run_processor)
    
#     print("Scheduler started. Scraping every 6 hours, processing 30 minutes later.")
    
#     while True:
#         schedule.run_pending()
#         time.sleep(60)  # Check every minute

# if __name__ == "__main__":
#     setup_scheduler()   