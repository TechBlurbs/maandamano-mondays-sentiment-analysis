#!/usr/bin/env python3
"""
Comprehensive Sentiment Analysis Report Generator
===============================================

This script generates a comprehensive report with visualizations and insights
for the Maandamano Mondays sentiment analysis project.
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
import os
import re

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
        # Handle both formats: [0.1 0.2 0.3] and [0.1, 0.2, 0.3]
        label_string = label_string.strip('[]')
        # Split by comma or whitespace
        if ',' in label_string:
            values = [float(val.strip()) for val in label_string.split(',')]
        else:
            values = [float(val) for val in label_string.split()]
        return values
    except:
        return [0.0, 0.0, 0.0]

def get_sentiment_class(probabilities):
    """Get sentiment class from probabilities [negative, neutral, positive]."""
    if not probabilities or len(probabilities) != 3:
        return "unknown"
    
    max_idx = probabilities.index(max(probabilities))
    classes = ["negative", "neutral", "positive"]
    return classes[max_idx]

def generate_html_report(data):
    """Generate an HTML report with visualizations."""
    
    # Process data for analysis
    sentiment_stats = analyze_sentiment_data(data)
    hashtag_stats = analyze_hashtag_data(data)
    user_stats = analyze_user_data(data)
    text_stats = analyze_text_data(data)
    economic_stats = analyze_economic_impact(data)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Maandamano Mondays Sentiment Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card h3 {{
            margin-top: 0;
            color: #333;
        }}
        .sentiment-bar {{
            height: 30px;
            background: #f0f0f0;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .sentiment-negative {{
            background: #ff4444;
            height: 100%;
            float: left;
        }}
        .sentiment-neutral {{
            background: #888888;
            height: 100%;
            float: left;
        }}
        .sentiment-positive {{
            background: #44ff44;
            height: 100%;
            float: left;
        }}
        .chart-container {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .bar-chart {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .bar-item {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .bar-label {{
            width: 150px;
            font-size: 12px;
        }}
        .bar {{
            height: 20px;
            background: #4CAF50;
            min-width: 20px;
            border-radius: 3px;
        }}
        .bar-value {{
            font-size: 12px;
            font-weight: bold;
        }}
        .recommendations {{
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 20px;
            margin: 20px 0;
        }}
        .alert {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .alert.danger {{
            background: #f8d7da;
            border-color: #f5c6cb;
            color: #721c24;
        }}
        .alert.success {{
            background: #d4edda;
            border-color: #c3e6cb;
            color: #155724;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .footer {{
            margin-top: 50px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            text-align: center;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Maandamano Mondays Sentiment Analysis</h1>
        <p>Comprehensive Report on Public Sentiment and Economic Impact</p>
        <p><strong>Generated on:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="summary-cards">
        <div class="card">
            <h3>📊 Dataset Overview</h3>
            <p><strong>Total Tweets:</strong> {len(data):,}</p>
            <p><strong>Unique Users:</strong> {sentiment_stats['unique_users']:,}</p>
            <p><strong>Unique Hashtags:</strong> {len(hashtag_stats):,}</p>
            <p><strong>Avg Confidence:</strong> {sentiment_stats['avg_confidence']:.3f}</p>
        </div>
        
        <div class="card">
            <h3>😊 Sentiment Distribution</h3>
            <div class="sentiment-bar">
                <div class="sentiment-negative" style="width: {sentiment_stats['negative_pct']:.1f}%"></div>
                <div class="sentiment-neutral" style="width: {sentiment_stats['neutral_pct']:.1f}%"></div>
                <div class="sentiment-positive" style="width: {sentiment_stats['positive_pct']:.1f}%"></div>
            </div>
            <p>🔴 Negative: {sentiment_stats['negative_pct']:.1f}%</p>
            <p>⚪ Neutral: {sentiment_stats['neutral_pct']:.1f}%</p>
            <p>🟢 Positive: {sentiment_stats['positive_pct']:.1f}%</p>
        </div>
        
        <div class="card">
            <h3>📈 Key Metrics</h3>
            <p><strong>Dominant Sentiment:</strong> {sentiment_stats['dominant_sentiment']}</p>
            <p><strong>Concern Level:</strong> {sentiment_stats['concern_level']}</p>
            <p><strong>Economic Tweets:</strong> {economic_stats['economic_tweet_count']}</p>
            <p><strong>Coverage:</strong> {(len(data)/2584*100):.1f}% of raw data</p>
        </div>
        
        <div class="card">
            <h3>👥 User Engagement</h3>
            <p><strong>Most Active:</strong> @{user_stats['most_active_user']}</p>
            <p><strong>Multi-tweet Users:</strong> {user_stats['multi_tweet_users']}</p>
            <p><strong>Avg Tweets/User:</strong> {user_stats['avg_tweets_per_user']:.1f}</p>
            <p><strong>Top Hashtag:</strong> #{list(hashtag_stats.keys())[0] if hashtag_stats else 'N/A'}</p>
        </div>
    </div>

    {generate_alert_section(sentiment_stats)}

    <div class="chart-container">
        <h3>🏷️ Top Hashtags by Volume</h3>
        <div class="bar-chart">
            {generate_hashtag_chart(hashtag_stats)}
        </div>
    </div>

    <div class="chart-container">
        <h3>👤 Most Active Users</h3>
        <div class="bar-chart">
            {generate_user_chart(user_stats)}
        </div>
    </div>

    <div class="chart-container">
        <h3>💼 Economic Impact Analysis</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
            <tr>
                <td>Business-related tweets</td>
                <td>{economic_stats['business_tweets']}</td>
                <td>{economic_stats['business_pct']:.1f}%</td>
            </tr>
            <tr>
                <td>Economic-related tweets</td>
                <td>{economic_stats['economic_tweets']}</td>
                <td>{economic_stats['economic_pct']:.1f}%</td>
            </tr>
            <tr>
                <td>Protest impact mentions</td>
                <td>{economic_stats['protest_impact_tweets']}</td>
                <td>{economic_stats['protest_impact_pct']:.1f}%</td>
            </tr>
        </table>
    </div>

    <div class="chart-container">
        <h3>🔤 Key Terms by Sentiment</h3>
        {generate_text_analysis_section(text_stats)}
    </div>

    {generate_recommendations_section(sentiment_stats, economic_stats)}

    <div class="footer">
        <p>Report generated using sentiment analysis on {len(data):,} tweets</p>
        <p>Methodology: Cardiff NLP Twitter XLM-RoBERTa sentiment classification</p>
        <p>For questions or additional analysis, contact the research team</p>
    </div>
</body>
</html>
"""
    
    return html_content

def analyze_sentiment_data(data):
    """Analyze sentiment distribution and statistics."""
    sentiment_counts = Counter()
    confidence_scores = []
    usernames = set()
    
    for row in data:
        usernames.add(row.get('username', ''))
        if 'labels' in row:
            probs = parse_sentiment_labels(row['labels'])
            sentiment = get_sentiment_class(probs)
            sentiment_counts[sentiment] += 1
            confidence_scores.append(max(probs) if probs else 0)
    
    total = sum(sentiment_counts.values())
    
    neg_pct = (sentiment_counts.get('negative', 0) / total * 100) if total > 0 else 0
    neu_pct = (sentiment_counts.get('neutral', 0) / total * 100) if total > 0 else 0
    pos_pct = (sentiment_counts.get('positive', 0) / total * 100) if total > 0 else 0
    
    # Determine dominant sentiment and concern level
    if neg_pct > 50:
        dominant_sentiment = "NEGATIVE"
        concern_level = "HIGH CONCERN"
    elif pos_pct > 50:
        dominant_sentiment = "POSITIVE" 
        concern_level = "LOW CONCERN"
    else:
        dominant_sentiment = "MIXED"
        concern_level = "MODERATE CONCERN"
    
    return {
        'sentiment_counts': sentiment_counts,
        'negative_pct': neg_pct,
        'neutral_pct': neu_pct,
        'positive_pct': pos_pct,
        'avg_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
        'unique_users': len(usernames),
        'dominant_sentiment': dominant_sentiment,
        'concern_level': concern_level
    }

def analyze_hashtag_data(data):
    """Analyze hashtag patterns and sentiment."""
    hashtag_sentiment = defaultdict(lambda: {"negative": 0, "neutral": 0, "positive": 0})
    
    for row in data:
        if 'labels' in row and 'extract_hashtags' in row:
            probs = parse_sentiment_labels(row['labels'])
            sentiment = get_sentiment_class(probs)
            
            hashtags_text = row.get('extract_hashtags', '')
            hashtags = hashtags_text.split() if hashtags_text else []
            
            for hashtag in hashtags:
                hashtag_sentiment[hashtag][sentiment] += 1
    
    # Sort by total volume
    hashtag_totals = {}
    for hashtag, sentiments in hashtag_sentiment.items():
        total = sum(sentiments.values())
        if total >= 5:  # Filter hashtags with at least 5 mentions
            hashtag_totals[hashtag] = total
    
    return dict(sorted(hashtag_totals.items(), key=lambda x: x[1], reverse=True))

def analyze_user_data(data):
    """Analyze user engagement patterns."""
    user_tweet_counts = Counter()
    
    for row in data:
        username = row.get('username', '').strip()
        if username:
            user_tweet_counts[username] += 1
    
    multi_tweet_users = len([user for user, count in user_tweet_counts.items() if count > 1])
    most_active_user = user_tweet_counts.most_common(1)[0][0] if user_tweet_counts else "None"
    avg_tweets_per_user = sum(user_tweet_counts.values()) / len(user_tweet_counts) if user_tweet_counts else 0
    
    return {
        'user_tweet_counts': user_tweet_counts,
        'multi_tweet_users': multi_tweet_users,
        'most_active_user': most_active_user,
        'avg_tweets_per_user': avg_tweets_per_user
    }

def analyze_text_data(data):
    """Analyze text patterns by sentiment."""
    sentiment_texts = {"negative": [], "neutral": [], "positive": []}
    
    for row in data:
        if 'labels' in row and 'lemmatized_text' in row:
            probs = parse_sentiment_labels(row['labels'])
            sentiment = get_sentiment_class(probs)
            
            text = row.get('lemmatized_text', '').lower()
            if text.strip():
                sentiment_texts[sentiment].append(text)
    
    # Extract top words for each sentiment
    sentiment_words = {}
    for sentiment, texts in sentiment_texts.items():
        if texts:
            word_counts = Counter()
            for text in texts:
                words = text.split()
                for word in words:
                    if len(word) > 3 and word.isalpha():
                        word_counts[word] += 1
            sentiment_words[sentiment] = dict(word_counts.most_common(10))
    
    return sentiment_words

def analyze_economic_impact(data):
    """Analyze economic impact mentions."""
    business_keywords = ['business', 'shop', 'duka', 'economy', 'money', 'work', 'job', 'income', 'trade', 'market']
    economic_keywords = ['cost', 'price', 'expensive', 'cheap', 'afford', 'salary', 'pay', 'buy', 'sell']
    protest_impact_keywords = ['close', 'closed', 'shutdown', 'block', 'blocked', 'stop', 'stopped', 'cancel']
    
    business_tweets = 0
    economic_tweets = 0
    protest_impact_tweets = 0
    economic_tweet_count = 0
    
    for row in data:
        text = row.get('lemmatized_text', '').lower()
        
        if any(keyword in text for keyword in business_keywords):
            business_tweets += 1
        if any(keyword in text for keyword in economic_keywords):
            economic_tweets += 1
        if any(keyword in text for keyword in protest_impact_keywords):
            protest_impact_tweets += 1
        if any(keyword in text for keyword in business_keywords + economic_keywords + protest_impact_keywords):
            economic_tweet_count += 1
    
    total = len(data)
    
    return {
        'business_tweets': business_tweets,
        'business_pct': (business_tweets / total * 100) if total > 0 else 0,
        'economic_tweets': economic_tweets,
        'economic_pct': (economic_tweets / total * 100) if total > 0 else 0,
        'protest_impact_tweets': protest_impact_tweets,
        'protest_impact_pct': (protest_impact_tweets / total * 100) if total > 0 else 0,
        'economic_tweet_count': economic_tweet_count
    }

def generate_alert_section(sentiment_stats):
    """Generate alert section based on sentiment analysis."""
    alert_class = "danger" if sentiment_stats['negative_pct'] > 60 else "success" if sentiment_stats['positive_pct'] > 40 else "alert"
    
    if sentiment_stats['negative_pct'] > 60:
        message = f"⚠️ CRITICAL: {sentiment_stats['negative_pct']:.1f}% negative sentiment indicates severe public concern requiring immediate attention"
    elif sentiment_stats['negative_pct'] > 40:
        message = f"⚠️ WARNING: {sentiment_stats['negative_pct']:.1f}% negative sentiment suggests significant public dissatisfaction"
    else:
        message = f"✅ STABLE: Sentiment distribution appears balanced with manageable concern levels"
    
    return f'<div class="alert {alert_class}">{message}</div>'

def generate_hashtag_chart(hashtag_stats):
    """Generate hashtag chart HTML."""
    chart_html = ""
    max_count = max(hashtag_stats.values()) if hashtag_stats else 1
    
    for hashtag, count in list(hashtag_stats.items())[:10]:
        width = (count / max_count) * 100
        chart_html += f"""
        <div class="bar-item">
            <div class="bar-label">#{hashtag}</div>
            <div class="bar" style="width: {width}%; background: #4CAF50;"></div>
            <div class="bar-value">{count}</div>
        </div>
        """
    
    return chart_html

def generate_user_chart(user_stats):
    """Generate user activity chart HTML."""
    chart_html = ""
    top_users = user_stats['user_tweet_counts'].most_common(10)
    max_count = top_users[0][1] if top_users else 1
    
    for username, count in top_users:
        width = (count / max_count) * 100
        chart_html += f"""
        <div class="bar-item">
            <div class="bar-label">@{username}</div>
            <div class="bar" style="width: {width}%; background: #2196F3;"></div>
            <div class="bar-value">{count}</div>
        </div>
        """
    
    return chart_html

def generate_text_analysis_section(text_stats):
    """Generate text analysis section."""
    section_html = ""
    colors = {'negative': '#ff4444', 'neutral': '#888888', 'positive': '#44ff44'}
    
    for sentiment, words in text_stats.items():
        if words:
            section_html += f"<h4>{sentiment.capitalize()} Sentiment Key Terms</h4>"
            section_html += '<div class="bar-chart">'
            
            max_count = max(words.values()) if words else 1
            for word, count in words.items():
                width = (count / max_count) * 80
                section_html += f"""
                <div class="bar-item">
                    <div class="bar-label">{word}</div>
                    <div class="bar" style="width: {width}%; background: {colors[sentiment]};"></div>
                    <div class="bar-value">{count}</div>
                </div>
                """
            section_html += '</div>'
    
    return section_html

def generate_recommendations_section(sentiment_stats, economic_stats):
    """Generate recommendations based on analysis."""
    recommendations_html = '<div class="recommendations">'
    recommendations_html += '<h3>💡 Key Recommendations</h3>'
    
    if sentiment_stats['negative_pct'] > 60:
        recommendations_html += """
        <h4>🚨 Immediate Actions Required:</h4>
        <ul>
            <li>Address underlying issues causing widespread negative sentiment</li>
            <li>Implement emergency economic support for affected businesses</li>
            <li>Establish crisis communication channels with the public</li>
            <li>Consider policy adjustments to address core concerns</li>
        </ul>
        """
    elif sentiment_stats['negative_pct'] > 40:
        recommendations_html += """
        <h4>⚠️ Proactive Measures Needed:</h4>
        <ul>
            <li>Engage in dialogue with protest organizers and civil society</li>
            <li>Monitor economic impact on small businesses</li>
            <li>Improve communication about government policies</li>
            <li>Consider targeted interventions for specific concerns</li>
        </ul>
        """
    else:
        recommendations_html += """
        <h4>✅ Monitoring and Maintenance:</h4>
        <ul>
            <li>Continue monitoring sentiment trends for early warning signs</li>
            <li>Maintain open communication channels</li>
            <li>Support economic recovery in affected areas</li>
            <li>Build on positive sentiment to strengthen public confidence</li>
        </ul>
        """
    
    recommendations_html += """
    <h4>📊 Long-term Strategic Actions:</h4>
    <ul>
        <li>Develop sentiment monitoring system for early detection of issues</li>
        <li>Create economic impact assessment framework for future protests</li>
        <li>Establish regular stakeholder engagement mechanisms</li>
        <li>Implement transparency measures to build public trust</li>
    </ul>
    """
    
    recommendations_html += '</div>'
    return recommendations_html

def main():
    """Main function to generate comprehensive report."""
    print("Generating Comprehensive Sentiment Analysis Report")
    print("=" * 60)
    
    # Load data
    data = read_csv_file('data/extended_labeled_tweets.csv')
    if not data:
        data = read_csv_file('data/labeled_tweets.csv')
    
    if not data:
        print("No labeled data found. Please ensure labeled data exists.")
        return
    
    # Generate HTML report
    html_content = generate_html_report(data)
    
    # Save report
    report_filename = f"maandamano_sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    try:
        with open(report_filename, 'w', encoding='utf-8') as file:
            file.write(html_content)
        
        print(f"✅ Comprehensive report generated: {report_filename}")
        print(f"📊 Report includes analysis of {len(data):,} tweets")
        print(f"🌐 Open the HTML file in your browser to view the interactive report")
        
    except Exception as e:
        print(f"Error generating report: {e}")

if __name__ == "__main__":
    main()