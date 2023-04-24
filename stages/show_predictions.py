from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

from pyspark.sql.functions import col


def main():
    conf = (SparkConf().setMaster('local[4]'))

    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    df_predictions = sqlContext.read.csv("predictions/lr-model/*.csv", header=True, inferSchema=True)

    df_predictions.sort(col("P1").desc()).show(100)

    print(df_predictions.select('id').distinct().count())


if __name__ == '__main__':
    main()
