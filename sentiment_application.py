#!/usr/bin/env python3
"""
Sentiment Analysis Application for Maandamano Mondays Data
=========================================================

This script performs comprehensive sentiment analysis on the Maandamano Mondays dataset,
including exploratory data analysis, visualizations, and reporting.

Features:
- Process remaining tweets with sentiment analysis
- Comprehensive EDA and insights
- Visualizations and reporting
- Economic impact analysis
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
import os


def read_csv_file(filepath):
    """Read CSV file and return data as list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        print(f"Successfully loaded {len(data)} records from {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return []
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []


def parse_sentiment_labels(label_string):
    """Parse sentiment label string into probabilities."""
    try:
        # Remove brackets and split by whitespace
        label_string = label_string.strip('[]')
        values = label_string.split()
        return [float(val) for val in values]
    except:
        return [0.0, 0.0, 0.0]


def get_sentiment_class(probabilities):
    """Get sentiment class from probabilities [negative, neutral, positive]."""
    if not probabilities or len(probabilities) != 3:
        return "unknown"
    
    max_idx = probabilities.index(max(probabilities))
    classes = ["negative", "neutral", "positive"]
    return classes[max_idx]


def analyze_sentiment_distribution(data):
    """Analyze sentiment distribution in the data."""
    sentiment_counts = Counter()
    confidence_scores = []
    
    for row in data:
        if 'labels' in row:
            probs = parse_sentiment_labels(row['labels'])
            sentiment = get_sentiment_class(probs)
            sentiment_counts[sentiment] += 1
            confidence_scores.append(max(probs) if probs else 0)
    
    total = sum(sentiment_counts.values())
    
    print("\n=== SENTIMENT DISTRIBUTION ===")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{sentiment.capitalize()}: {count} tweets ({percentage:.1f}%)")
    
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    print(f"\nAverage Confidence Score: {avg_confidence:.3f}")
    
    return sentiment_counts, confidence_scores


def analyze_hashtags_by_sentiment(data):
    """Analyze hashtag usage patterns by sentiment."""
    hashtag_sentiment = defaultdict(lambda: {"negative": 0, "neutral": 0, "positive": 0})
    
    for row in data:
        if 'labels' in row and 'extract_hashtags' in row:
            probs = parse_sentiment_labels(row['labels'])
            sentiment = get_sentiment_class(probs)
            
            # Extract hashtags
            hashtags_text = row.get('extract_hashtags', '')
            hashtags = hashtags_text.split() if hashtags_text else []
            
            for hashtag in hashtags:
                hashtag_sentiment[hashtag][sentiment] += 1
    
    print("\n=== HASHTAG SENTIMENT ANALYSIS ===")
    for hashtag, sentiments in sorted(hashtag_sentiment.items(), 
                                     key=lambda x: sum(x[1].values()), reverse=True)[:10]:
        total = sum(sentiments.values())
        if total >= 3:  # Only show hashtags with at least 3 mentions
            neg_pct = sentiments["negative"] / total * 100
            neu_pct = sentiments["neutral"] / total * 100
            pos_pct = sentiments["positive"] / total * 100
            print(f"#{hashtag}: Neg:{neg_pct:.1f}% Neu:{neu_pct:.1f}% Pos:{pos_pct:.1f}% (n={total})")
    
    return hashtag_sentiment


def analyze_text_patterns(data):
    """Analyze text patterns and key terms by sentiment."""
    sentiment_texts = {"negative": [], "neutral": [], "positive": []}
    
    for row in data:
        if 'labels' in row and 'lemmatized_text' in row:
            probs = parse_sentiment_labels(row['labels'])
            sentiment = get_sentiment_class(probs)
            
            text = row.get('lemmatized_text', '').lower()
            if text.strip():
                sentiment_texts[sentiment].append(text)
    
    print("\n=== TEXT PATTERN ANALYSIS ===")
    for sentiment, texts in sentiment_texts.items():
        if texts:
            # Count word frequency
            word_counts = Counter()
            for text in texts:
                words = text.split()
                for word in words:
                    if len(word) > 3 and word.isalpha():  # Filter short words and non-alphabetic
                        word_counts[word] += 1
            
            print(f"\nTop words in {sentiment} tweets:")
            for word, count in word_counts.most_common(10):
                print(f"  {word}: {count}")
    
    return sentiment_texts


def analyze_user_patterns(data):
    """Analyze user engagement patterns."""
    user_sentiments = defaultdict(list)
    
    for row in data:
        if 'labels' in row and 'username' in row:
            probs = parse_sentiment_labels(row['labels'])
            sentiment = get_sentiment_class(probs)
            username = row.get('username', '').strip()
            
            if username:
                user_sentiments[username].append(sentiment)
    
    print("\n=== USER ENGAGEMENT PATTERNS ===")
    
    # Users with multiple tweets
    multi_tweet_users = {user: sentiments for user, sentiments in user_sentiments.items() 
                        if len(sentiments) > 1}
    
    print(f"Total unique users: {len(user_sentiments)}")
    print(f"Users with multiple tweets: {len(multi_tweet_users)}")
    
    # Most active users
    print("\nMost active users:")
    for user, sentiments in sorted(user_sentiments.items(), 
                                  key=lambda x: len(x[1]), reverse=True)[:10]:
        sentiment_counts = Counter(sentiments)
        print(f"  @{user}: {len(sentiments)} tweets - "
              f"Neg:{sentiment_counts['negative']} "
              f"Neu:{sentiment_counts['neutral']} "
              f"Pos:{sentiment_counts['positive']}")
    
    return user_sentiments


def generate_insights_report(sentiment_counts, hashtag_sentiment, sentiment_texts, user_sentiments):
    """Generate insights and economic impact analysis."""
    total_tweets = sum(sentiment_counts.values())
    
    print("\n" + "="*50)
    print("MAANDAMANO MONDAYS SENTIMENT ANALYSIS REPORT")
    print("="*50)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"• Total analyzed tweets: {total_tweets}")
    print(f"• Unique users: {len(user_sentiments)}")
    print(f"• Unique hashtags: {len(hashtag_sentiment)}")
    
    print(f"\n😊 SENTIMENT SUMMARY:")
    if total_tweets > 0:
        neg_pct = sentiment_counts.get('negative', 0) / total_tweets * 100
        neu_pct = sentiment_counts.get('neutral', 0) / total_tweets * 100
        pos_pct = sentiment_counts.get('positive', 0) / total_tweets * 100
        
        print(f"• Negative sentiment: {neg_pct:.1f}%")
        print(f"• Neutral sentiment: {neu_pct:.1f}%")
        print(f"• Positive sentiment: {pos_pct:.1f}%")
        
        if neg_pct > 50:
            dominant_sentiment = "NEGATIVE"
            impact = "HIGH CONCERN"
        elif pos_pct > 50:
            dominant_sentiment = "POSITIVE"
            impact = "LOW CONCERN"
        else:
            dominant_sentiment = "MIXED"
            impact = "MODERATE CONCERN"
        
        print(f"\n🎯 KEY FINDINGS:")
        print(f"• Dominant sentiment: {dominant_sentiment}")
        print(f"• Economic concern level: {impact}")
        
        # Hashtag insights
        maandamano_hashtags = [h for h in hashtag_sentiment.keys() 
                              if 'maandamano' in h.lower()]
        if maandamano_hashtags:
            print(f"• Maandamano-related hashtags found: {len(maandamano_hashtags)}")
        
        # Business impact insights
        business_keywords = ['business', 'shop', 'duka', 'economy', 'money', 'work', 'job']
        business_mentions = 0
        for texts in sentiment_texts.values():
            for text in texts:
                if any(keyword in text.lower() for keyword in business_keywords):
                    business_mentions += 1
        
        if business_mentions > 0:
            print(f"• Tweets mentioning business/economic terms: {business_mentions}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if sentiment_counts.get('negative', 0) > sentiment_counts.get('positive', 0):
        print("• High negative sentiment suggests significant public concern")
        print("• Recommend addressing underlying issues to improve public sentiment")
        print("• Monitor economic impact on businesses, especially in affected areas")
    else:
        print("• Sentiment appears balanced or positive")
        print("• Continue monitoring for changes in public opinion")
    
    print("\n📈 ECONOMIC IMPLICATIONS:")
    print("• Business closures during protest days likely impact local economy")
    print("• Negative sentiment may affect consumer confidence")
    print("• Sustained protests could have cumulative economic effects")


def create_simple_visualizations(sentiment_counts, hashtag_sentiment):
    """Create simple text-based visualizations."""
    print("\n=== SENTIMENT DISTRIBUTION CHART ===")
    total = sum(sentiment_counts.values())
    
    if total > 0:
        for sentiment, count in sentiment_counts.items():
            percentage = count / total * 100
            bar_length = int(percentage / 2)  # Scale for display
            bar = "█" * bar_length
            print(f"{sentiment.capitalize():>8}: {bar} {percentage:.1f}% ({count})")
    
    print("\n=== TOP HASHTAGS BY VOLUME ===")
    hashtag_totals = {hashtag: sum(sentiments.values()) 
                     for hashtag, sentiments in hashtag_sentiment.items()}
    
    for hashtag, total in sorted(hashtag_totals.items(), 
                                key=lambda x: x[1], reverse=True)[:10]:
        bar_length = min(total, 20)  # Cap at 20 for display
        bar = "▓" * bar_length
        print(f"#{hashtag:>15}: {bar} ({total})")


def process_remaining_tweets():
    """Process remaining tweets that haven't been labeled yet."""
    print("\n=== PROCESSING REMAINING TWEETS ===")
    
    # Load raw tweets
    raw_tweets = read_csv_file('data/tweets.csv')
    labeled_tweets = read_csv_file('data/labeled_tweets.csv')
    
    if not raw_tweets:
        print("No raw tweets found to process")
        return
    
    labeled_usernames = set()
    labeled_texts = set()
    
    # Get already processed tweets
    for row in labeled_tweets:
        if 'username' in row:
            labeled_usernames.add(row['username'])
        if 'lemmatized_text' in row:
            labeled_texts.add(row['lemmatized_text'])
    
    # Find unprocessed tweets
    unprocessed = []
    for tweet in raw_tweets:
        username = tweet.get('Username', '')
        text = tweet.get('Text', '')
        
        # Simple check if not already processed
        if username not in labeled_usernames or text not in labeled_texts:
            unprocessed.append(tweet)
    
    print(f"Found {len(unprocessed)} potentially unprocessed tweets")
    print("Note: Full sentiment labeling requires the transformer model")
    print("      Consider running the add_labels.py script to process remaining tweets")
    
    return unprocessed


def main():
    """Main function to run the sentiment analysis application."""
    print("Maandamano Mondays Sentiment Analysis Application")
    print("=" * 50)
    
    # Try to load extended dataset first, fall back to original
    labeled_data = read_csv_file('data/extended_labeled_tweets.csv')
    if not labeled_data:
        labeled_data = read_csv_file('data/labeled_tweets.csv')
    
    if not labeled_data:
        print("No labeled data found. Please ensure labeled_tweets.csv exists.")
        return
    
    # Perform analysis
    sentiment_counts, confidence_scores = analyze_sentiment_distribution(labeled_data)
    hashtag_sentiment = analyze_hashtags_by_sentiment(labeled_data)
    sentiment_texts = analyze_text_patterns(labeled_data)
    user_sentiments = analyze_user_patterns(labeled_data)
    
    # Generate visualizations
    create_simple_visualizations(sentiment_counts, hashtag_sentiment)
    
    # Generate insights report
    generate_insights_report(sentiment_counts, hashtag_sentiment, sentiment_texts, user_sentiments)
    
    # Check for remaining tweets to process
    process_remaining_tweets()
    
    print(f"\n✅ Analysis complete! Check the results above for insights.")


if __name__ == "__main__":
    main()