import asyncio
import yaml
import os
from telethon.sync import TelegramClient
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
import pandas as pd
import json
from datetime import datetime
from src.preprocessing.preprocess import preprocess_amharic_text

async def connect_to_telegram(config_path):
    """Initialize and connect to Telegram API using credentials from config.yaml."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        api_id = config['telegram']['api_id']
        api_hash = config['telegram']['api_hash']
        phone = config['telegram']['phone']
        
        client = TelegramClient('session_name', api_id, api_hash)
        await client.start(phone=phone)
        print("Successfully connected to Telegram API")
        return client
    except Exception as e:
        print(f"Failed to connect to Telegram: {e}")
        raise

async def scrape_telegram_channel(client, channel_username, output_dir_raw, output_dir_processed, limit=100):
    """Scrape messages from a Telegram channel and preprocess text."""
    os.makedirs(output_dir_raw, exist_ok=True)
    os.makedirs(output_dir_processed, exist_ok=True)
    
    messages_data = []
    try:
        async for message in client.iter_messages(channel_username, limit=limit):
            msg_data = {
                'channel': channel_username,
                'message_id': message.id,
                'date': message.date.isoformat(),
                'text': message.text or '',
                'sender_id': message.sender_id,
                'views': message.views if message.views else 0,
                'image_path': None,
                'doc_path': None
            }
            
            # Handle images
            if isinstance(message.media, MessageMediaPhoto):
                image_path = os.path.join(output_dir_raw, 'images', f"{channel_username}_{message.id}.jpg")
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                await message.download_media(file=image_path)
                msg_data['image_path'] = image_path
            
            # Handle documents
            if isinstance(message.media, MessageMediaDocument):
                doc_path = os.path.join(output_dir_raw, 'documents', f"{channel_username}_{message.id}")
                os.makedirs(os.path.dirname(doc_path), exist_ok=True)
                await message.download_media(file=doc_path)
                msg_data['doc_path'] = doc_path
            
            # Preprocess text
            msg_data['processed_text'] = preprocess_amharic_text(msg_data['text'])
            messages_data.append(msg_data)
        
        # Save raw data as JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_output_path = os.path.join(output_dir_raw, f"{channel_username[1:]}_{timestamp}.json")
        with open(raw_output_path, 'w', encoding='utf-8') as f:
            json.dump(messages_data, f, ensure_ascii=False, indent=2)
        print(f"Saved raw data to {raw_output_path}")
        
        # Save processed data as CSV
        df = pd.DataFrame(messages_data)
        processed_output_path = os.path.join(output_dir_processed, f"{channel_username[1:]}_{timestamp}.csv")
        df.to_csv(processed_output_path, index=False, encoding='utf-8')
        print(f"Saved processed data to {processed_output_path}")
        
        return messages_data
    except Exception as e:
        print(f"Error scraping {channel_username}: {e}")
        return []

async def main():
    """Scrape multiple Telegram channels."""
    config_path = 'config.yaml'
    output_dir_raw = 'data/raw'
    output_dir_processed = 'data/processed'
    
    # List of Ethiopian e-commerce channels (replace with actual channels)
    channels = [
        '@Shageronlinestore',
        '@EthioEcommerce1',
        '@EthioMartSales',
        '@AddisShopping',
        '@EthioOnlineBazaar'
    ]
    
    client = await connect_to_telegram(config_path)
    async with client:
        for channel in channels:
            print(f"Scraping {channel}...")
            await scrape_telegram_channel(client, channel, output_dir_raw, output_dir_processed, limit=100)

if __name__ == "__main__":
    asyncio.run(main())