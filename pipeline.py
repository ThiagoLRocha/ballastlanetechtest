
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import ast
import json

def create_spark_session():
    """Creating Spark session"""
    spark = SparkSession.builder \
        .appName("CafeRewardsDataPipeline") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark

def load_raw_data(spark):
    """
    Loading data from CSV files 
    """
    print("=== LOADING RAW DATA ===")
    
    # Loading offers data
    offers_df = spark.read.csv('csv_PATH', header=True, inferSchema=True)
    print(f"Offers loaded: {offers_df.count()} rows")
    
    # Loading customers data
    customers_df = spark.read.csv('csv_PATH', header=True, inferSchema=True)
    print(f"Customers loaded: {customers_df.count()} rows")
    
    # Loading event data
    events_df = spark.read.csv('csv_PATH', header=True, inferSchema=True)
    print(f"Events loaded: {events_df.count()} rows")
    
    return offers_df, customers_df, events_df

def clean_and_transform_data(spark, offers_df, customers_df, events_df):
    """
    Cleaning and transformin data 
    """
    print("\n=== CLEANING AND TRANSFORMING DATA ===")
    
    # 1. Cleaning Offers Data
    print("1. Cleaning offers data...")
    
    # Parsing channels column from string to array
    def parse_channels_udf(channels_str):
        try:
            return ast.literal_eval(channels_str) if channels_str else []
        except:
            return []
    
    parse_channels = udf(parse_channels_udf, ArrayType(StringType()))
    
    offers_clean = offers_df.withColumn("channels_array", parse_channels(col("channels"))) \
                            .drop("channels") \
                            .withColumnRenamed("channels_array", "channels")
    
    print(f"Offers cleaned: {offers_clean.count()} rows")
    
    # 2. Cleaning Customers Data
    print("2. Cleaning customers data...")
    
    # Filtering out invalid customers
    customers_clean = customers_df.filter(col("age") != 118) \
                                 .filter(col("gender").isNotNull()) \
                                 .filter(col("income").isNotNull())
    
    # Converting became_member_on date format
    customers_clean = customers_clean.withColumn(
        "became_member_date", 
        to_date(col("became_member_on").cast("string"), "yyyyMMdd")
    )
    
    print(f"Customers cleaned: {customers_clean.count()} rows (removed invalid records)")
    
    # 3. Cleaning Events Data
    print("3. Cleaning events data...")
    
    # Parsing value column based on event type
    def parse_value_udf(event_type, value_str):
        try:
            parsed = ast.literal_eval(value_str) if value_str else {}
            
            if event_type == 'transaction':
                return {'amount': parsed.get('amount', 0.0)}
            elif event_type in ['offer received', 'offer viewed']:
                return {'offer_id': parsed.get('offer id', '')}
            elif event_type == 'offer completed':
                return {
                    'offer_id': parsed.get('offer_id', ''),
                    'reward': parsed.get('reward', 0)
                }
            else:
                return parsed
        except:
            return {}
    
    parse_value = udf(parse_value_udf, MapType(StringType(), StringType()))
    
    events_clean = events_df.withColumn("parsed_value", parse_value(col("event"), col("value"))) \
                           .drop("value")
    
    # Extracting columns based on event type
    events_clean = events_clean.withColumn("offer_id", 
                                         when(col("event").isin(["offer received", "offer viewed", "offer completed"]),
                                              col("parsed_value").getItem("offer_id")).otherwise(None)) \
                             .withColumn("transaction_amount",
                                       when(col("event") == "transaction",
                                            col("parsed_value").getItem("amount").cast("double")).otherwise(None)) \
                             .withColumn("reward_earned",
                                       when(col("event") == "offer completed",
                                            col("parsed_value").getItem("reward").cast("double")).otherwise(None))
    
    # Filtering events for valid customers only
    valid_customers = customers_clean.select("customer_id").distinct()
    events_clean = events_clean.join(valid_customers, "customer_id", "inner")
    
    print(f"Events cleaned: {events_clean.count()} rows")
    
    return offers_clean, customers_clean, events_clean

def create_refined_data(spark, offers_clean, customers_clean, events_clean):
    """
    Create aggregated and refined data (Refined Layer)
    """
    print("\n=== CREATING REFINED DATA ===")
    
    # 1. Customer Summary
    print("1. Creating customer summary...")
    
    customer_transactions = events_clean.filter(col("event") == "transaction") \
                                       .groupBy("customer_id") \
                                       .agg(
                                           count("*").alias("total_transactions"),
                                           sum("transaction_amount").alias("total_spent"),
                                           avg("transaction_amount").alias("avg_transaction_amount")
                                       )
    
    customer_offers = events_clean.filter(col("event").isin(["offer received", "offer viewed", "offer completed"])) \
                                 .groupBy("customer_id") \
                                 .agg(
                                     count_distinct(when(col("event") == "offer received", col("offer_id"))).alias("offers_received"),
                                     count_distinct(when(col("event") == "offer viewed", col("offer_id"))).alias("offers_viewed"),
                                     count_distinct(when(col("event") == "offer completed", col("offer_id"))).alias("offers_completed"),
                                     sum(when(col("event") == "offer completed", col("reward_earned"))).alias("total_rewards_earned")
                                 )
    
    customer_summary = customers_clean.join(customer_transactions, "customer_id", "left") \
                                     .join(customer_offers, "customer_id", "left") \
                                     .fillna(0, ["total_transactions", "total_spent", "avg_transaction_amount",
                                               "offers_received", "offers_viewed", "offers_completed", "total_rewards_earned"])
    
    print(f"Customer summary created: {customer_summary.count()} rows")
    
    # 2. Offer Performance Analysis
    print("2. Creating offer performance analysis...")
    
    offer_stats = events_clean.filter(col("event").isin(["offer received", "offer viewed", "offer completed"])) \
                             .groupBy("offer_id") \
                             .agg(
                                 count_distinct(when(col("event") == "offer received", col("customer_id"))).alias("customers_received"),
                                 count_distinct(when(col("event") == "offer viewed", col("customer_id"))).alias("customers_viewed"),
                                 count_distinct(when(col("event") == "offer completed", col("customer_id"))).alias("customers_completed")
                             )
    
    offer_performance = offers_clean.join(offer_stats, offers_clean.offer_id == offer_stats.offer_id, "left") \
                                   .drop(offer_stats.offer_id) \
                                   .fillna(0, ["customers_received", "customers_viewed", "customers_completed"]) \
                                   .withColumn("completion_rate", 
                                             when(col("customers_received") > 0, 
                                                  col("customers_completed") / col("customers_received")).otherwise(0)) \
                                   .withColumn("view_rate",
                                             when(col("customers_received") > 0,
                                                  col("customers_viewed") / col("customers_received")).otherwise(0))
    
    print(f"Offer performance created: {offer_performance.count()} rows")
    
    # 3. Channel Effectiveness Analysis
    print("3. Creating channel effectiveness analysis...")
    
    # Explode channels array to analyze each channel separately
    offers_channels = offers_clean.select("offer_id", "offer_type", explode("channels").alias("channel"))
    
    channel_performance = offers_channels.join(offer_stats, "offer_id", "inner") \
                                        .groupBy("channel") \
                                        .agg(
                                            count("offer_id").alias("offers_sent"),
                                            sum("customers_received").alias("total_customers_received"),
                                            sum("customers_viewed").alias("total_customers_viewed"),
                                            sum("customers_completed").alias("total_customers_completed")
                                        ) \
                                        .withColumn("channel_completion_rate",
                                                  when(col("total_customers_received") > 0,
                                                       col("total_customers_completed") / col("total_customers_received")).otherwise(0)) \
                                        .withColumn("channel_view_rate",
                                                  when(col("total_customers_received") > 0,
                                                       col("total_customers_viewed") / col("total_customers_received")).otherwise(0))
    
    print(f"Channel performance created: {channel_performance.count()} rows")
    
    return customer_summary, offer_performance, channel_performance

def save_processed_data(customer_summary, offer_performance, channel_performance):
    """
    Saving processed data to files
    """
    print("\n=== SAVING PROCESSED DATA ===")
    
    # Converting to Pandas for easier saving and analysis
    customer_summary_pd = customer_summary.toPandas()
    offer_performance_pd = offer_performance.toPandas()
    channel_performance_pd = channel_performance.toPandas()
    
    # Save to CSV files
    customer_summary_pd.to_csv('/home/ubuntu/customer_summary.csv', index=False)
    offer_performance_pd.to_csv('/home/ubuntu/offer_performance.csv', index=False)
    channel_performance_pd.to_csv('/home/ubuntu/channel_performance.csv', index=False)
    
    print("Data saved to CSV files:")
    print("- customer_summary.csv")
    print("- offer_performance.csv")
    print("- channel_performance.csv")
    
    return customer_summary_pd, offer_performance_pd, channel_performance_pd

def main():
    """
    Main pipeline execution
    """
    print("Starting Cafe Rewards Data Processing Pipeline...")
    
    # Initializing Spark
    spark = create_spark_session()
    
    try:
        # Loading data
        offers_df, customers_df, events_df = load_raw_data(spark)
        
        # Cleaning and transforming data
        offers_clean, customers_clean, events_clean = clean_and_transform_data(
            spark, offers_df, customers_df, events_df
        )
        
        # Creating refined data
        customer_summary, offer_performance, channel_performance = create_refined_data(
            spark, offers_clean, customers_clean, events_clean
        )
        
        # Saving processed data
        customer_summary_pd, offer_performance_pd, channel_performance_pd = save_processed_data(
            customer_summary, offer_performance, channel_performance
        )
        
        print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        
        return customer_summary_pd, offer_performance_pd, channel_performance_pd
        
    finally:
        spark.stop()

if __name__ == "__main__":
    customer_summary_pd, offer_performance_pd, channel_performance_pd = main()