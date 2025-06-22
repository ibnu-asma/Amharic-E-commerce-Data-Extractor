
import asyncio
import os
import json
from telethon import TelegramClient
from datetime import datetime
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

async def connect_to_telegram():
    """Connect to Telegram using credentials from config.yaml."""
    client = TelegramClient(
        'session_name',
        config['telegram']['api_id'],
        config['telegram']['api_hash']
    )
    await client.start(phone=config['telegram']['phone'])
    if not await client.is_user_authorized():
        await client.send_code_request(config['telegram']['phone'])
        code = input('Enter the code you received: ')
        await client.sign_in(config['telegram']['phone'], code)
    logging.info("Connected to Telegram")
    return client

async def scrape_telegram_channel(client, channel, limit=100):
    """Scrape messages from a Telegram channel."""
    messages = []
    async for message in client.iter_messages(channel, limit=limit):
        if message.message or message.media:
            msg_data = {
                'channel': channel,
                'message_id': message.id,
                'date': message.date.isoformat(),
                'text': message.message or '',
                'sender_id': message.sender_id,
                'views': message.views or 0,
                'image_path': None,
                'doc_path': None
            }
            # Save media (images or documents)
            if message.media:
                media_path = f"data/raw/media/{channel}_{message.id}"
                os.makedirs(os.path.dirname(media_path), exist_ok=True)
                file_path = await client.download_media(message.media, media_path)
                if file_path and (file_path.endswith(('.jpg', '.png', '.jpeg'))):
                    msg_data['image_path'] = file_path
                elif file_path:
                    msg_data['doc_path'] = file_path
            messages.append(msg_data)
    return messages

def save_raw_data(messages, channel):
    """Save raw JSON data only."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('data/raw', exist_ok=True)
    
    # Save raw JSON
    raw_path = f"data/raw/{channel}_{timestamp}.json"
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved raw data to {raw_path}")

async def main():
    """Main function to scrape multiple channels."""
    client = await connect_to_telegram()
    channels = [
        '@ethio_market_place',
        '@helloomarketethiopia',
        '@jijietcom',
        
        '@ethiomarketo',
        '@AwasMart'
        
    ]
    for channel in channels:
        try:
            logging.info(f"Scraping channel: {channel}")
            messages = await scrape_telegram_channel(client, channel, limit=100)
            if messages:
                save_raw_data(messages, channel.replace('@', ''))
            else:
                logging.warning(f"No messages scraped from {channel}")
        except Exception as e:
            logging.error(f"Error scraping {channel}: {e}")
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())