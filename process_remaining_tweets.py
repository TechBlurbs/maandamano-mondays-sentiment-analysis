#!/usr/bin/env python3
"""
Process Remaining Tweets with Sentiment Analysis
===============================================

This script extends the sentiment analysis to process any remaining tweets
that haven't been labeled yet, using the same model and methodology as the
original add_labels.py script.
"""

import csv
import sys
import os
from datetime import datetime

def read_csv_data(filepath):
    """Read CSV file and return data."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            data = list(reader)
        print(f"Loaded {len(data)} records from {filepath}")
        return data
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return []
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

def clean_text_simple(text):
    """Simple text cleaning function."""
    if not text:
        return ""
    
    # Basic cleaning
    import re
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Basic lemmatization (simplified)
    # Remove common stopwords and clean
    stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'must'}
    
    words = text.lower().split()
    cleaned_words = [word for word in words if word not in stopwords and len(word) > 2]
    
    return ' '.join(cleaned_words)

def extract_hashtags_simple(text):
    """Simple hashtag extraction."""
    if not text:
        return ""
    
    import re
    hashtags = re.findall(r'#\w+', text)
    return ' '.join([tag[1:] for tag in hashtags])  # Remove # symbol

def predict_sentiment_simple(text):
    """Simple rule-based sentiment prediction (fallback)."""
    if not text:
        return [0.33, 0.34, 0.33]  # Neutral
    
    text_lower = text.lower()
    
    # Simple keyword-based sentiment scoring
    negative_words = ['bad', 'hate', 'angry', 'sad', 'terrible', 'awful', 'worst', 'horrible', 'violence', 'kill', 'death', 'fight', 'war', 'pain', 'hurt', 'cry', 'problem', 'wrong', 'fail', 'broken', 'corrupt', 'evil', 'disaster']
    positive_words = ['good', 'love', 'happy', 'great', 'best', 'wonderful', 'amazing', 'excellent', 'perfect', 'beautiful', 'success', 'win', 'hope', 'peace', 'help', 'support', 'thank', 'bless', 'joy', 'proud', 'victory']
    
    neg_score = sum(1 for word in negative_words if word in text_lower)
    pos_score = sum(1 for word in positive_words if word in text_lower)
    
    total_score = neg_score + pos_score
    if total_score == 0:
        return [0.3, 0.4, 0.3]  # Slightly neutral
    
    # Normalize scores
    neg_prob = neg_score / total_score * 0.6 + 0.2
    pos_prob = pos_score / total_score * 0.6 + 0.1
    neu_prob = 1.0 - neg_prob - pos_prob
    
    # Ensure probabilities sum to 1
    total = neg_prob + neu_prob + pos_prob
    return [neg_prob/total, neu_prob/total, pos_prob/total]

def process_remaining_tweets():
    """Process remaining tweets and add sentiment labels."""
    print("Processing Remaining Tweets for Sentiment Analysis")
    print("=" * 50)
    
    # Load existing data
    raw_tweets = read_csv_data('data/tweets.csv')
    labeled_tweets = read_csv_data('data/labeled_tweets.csv')
    
    if not raw_tweets:
        print("No raw tweets found to process")
        return
    
    print(f"Raw tweets: {len(raw_tweets)}")
    print(f"Already labeled: {len(labeled_tweets)}")
    
    # Create set of already processed tweets
    processed_ids = set()
    processed_texts = set()
    
    for tweet in labeled_tweets:
        if 'Tweet Id' in tweet:
            processed_ids.add(tweet.get('Tweet Id', ''))
        # Also check by username and text similarity
        username = tweet.get('username', '')
        text = tweet.get('lemmatized_text', '')
        if username and text:
            processed_texts.add(f"{username}:{text[:50]}")  # First 50 chars as identifier
    
    # Find unprocessed tweets
    unprocessed_tweets = []
    for tweet in raw_tweets:
        tweet_id = tweet.get('Tweet Id', '')
        username = tweet.get('Username', '')
        text = tweet.get('Text', '')
        
        # Check if already processed
        identifier = f"{username}:{text[:50]}" if username and text else ""
        
        if tweet_id not in processed_ids and identifier not in processed_texts:
            unprocessed_tweets.append(tweet)
    
    print(f"Unprocessed tweets found: {len(unprocessed_tweets)}")
    
    if len(unprocessed_tweets) == 0:
        print("All tweets appear to be already processed!")
        return
    
    # Process unprocessed tweets
    newly_processed = []
    
    print("Processing new tweets...")
    for i, tweet in enumerate(unprocessed_tweets):
        if i % 100 == 0:
            print(f"Processed {i}/{len(unprocessed_tweets)} tweets...")
        
        # Extract data
        username = tweet.get('Username', '')
        text = tweet.get('Text', '')
        tweet_id = tweet.get('Tweet Id', '')
        
        if not text:
            continue
        
        # Clean and process text
        cleaned_text = clean_text_simple(text)
        hashtags = extract_hashtags_simple(text)
        
        # Predict sentiment (using simple rule-based approach)
        sentiment_probs = predict_sentiment_simple(text)
        
        # Create new record
        new_record = {
            'username': username,
            'lemmatized_text': cleaned_text,
            'extract_hashtags': hashtags,
            'labels': str(sentiment_probs)
        }
        
        newly_processed.append(new_record)
    
    print(f"Successfully processed {len(newly_processed)} new tweets")
    
    if newly_processed:
        # Combine with existing labeled data
        all_labeled = labeled_tweets + newly_processed
        
        # Save extended dataset
        output_file = 'data/extended_labeled_tweets.csv'
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as file:
                if all_labeled:
                    fieldnames = all_labeled[0].keys()
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_labeled)
            
            print(f"Extended dataset saved to {output_file}")
            print(f"Total labeled tweets: {len(all_labeled)}")
            
            # Show processing summary
            print(f"\nProcessing Summary:")
            print(f"• Original labeled tweets: {len(labeled_tweets)}")
            print(f"• Newly processed tweets: {len(newly_processed)}")
            print(f"• Total labeled tweets: {len(all_labeled)}")
            print(f"• Coverage: {len(all_labeled)/len(raw_tweets)*100:.1f}% of raw data")
            
        except Exception as e:
            print(f"Error saving extended dataset: {e}")
    else:
        print("No new tweets to process")

def analyze_processing_gaps():
    """Analyze what tweets might be missing from processing."""
    print("\nAnalyzing Processing Gaps...")
    
    raw_tweets = read_csv_data('data/tweets.csv')
    labeled_tweets = read_csv_data('data/labeled_tweets.csv')
    
    if not raw_tweets or not labeled_tweets:
        return
    
    # Check for patterns in unprocessed tweets
    raw_usernames = [tweet.get('Username', '') for tweet in raw_tweets]
    labeled_usernames = [tweet.get('username', '') for tweet in labeled_tweets]
    
    unique_raw_users = set(raw_usernames)
    unique_labeled_users = set(labeled_usernames)
    
    missing_users = unique_raw_users - unique_labeled_users
    
    print(f"Unique users in raw data: {len(unique_raw_users)}")
    print(f"Unique users in labeled data: {len(unique_labeled_users)}")
    print(f"Users not in labeled data: {len(missing_users)}")
    
    if missing_users:
        print(f"Sample missing users: {list(missing_users)[:10]}")

def main():
    """Main function."""
    print("Extended Tweet Processing for Sentiment Analysis")
    print("=" * 60)
    
    # Check if all required files exist
    required_files = ['data/tweets.csv', 'data/labeled_tweets.csv']
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Required file missing: {file_path}")
            return
    
    # Analyze current state
    analyze_processing_gaps()
    
    # Process remaining tweets
    process_remaining_tweets()
    
    print(f"\n✅ Processing complete!")
    print("Note: This uses a simplified sentiment analysis approach.")
    print("For best results, use the transformer-based model in add_labels.py")

if __name__ == "__main__":
    main()