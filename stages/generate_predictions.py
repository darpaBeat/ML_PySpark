from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

from pyspark.ml import Pipeline, PipelineModel

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

from pyspark.ml.linalg import DenseVector

from pyspark.sql.functions import explode
from pyspark.sql.types import *

import pyspark.sql.functions as F
from pyspark.sql.functions import lit, udf


def encodeCategoricalFeatures(categorical_features):
    stages = []

    for categoricalCol in categorical_features:
        stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "_encoded")

        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()],
                                outputCols=[categoricalCol + "_vector"])

        stages += [stringIndexer, encoder]

    return stages


def assembleFeaturesVector(categorical_features, continuous_features):
    assemblerInputs = [c + "_vector" for c in categorical_features] + continuous_features

    return [VectorAssembler(inputCols=assemblerInputs, outputCol="features")]


def encodeLabel(label, new_label):
    return [StringIndexer(inputCol=label, outputCol=new_label)]


def transformFeatures(categorical_features, continuous_features, target):
    # STAGE 1: convert a string into one hot encoder
    stages = encodeCategoricalFeatures(categorical_features)

    # STAGE 2: encode income (label)
    stages += encodeLabel(target, "label")

    # STAGE 3: create vector with features
    stages += assembleFeaturesVector(categorical_features, continuous_features)

    return stages


def featuresPipeline(stages, df):
    pipeline = Pipeline(stages=stages)

    estimator = pipeline.fit(df)

    transformed_df = estimator.transform(df)

    return transformed_df


def valueToArray(value):
    size = 59
    return [value / x for x in range(2, size + 1)]


def explodeDF(df, feature, new_column):
    udfValueToArray = udf(valueToArray, ArrayType(FloatType()))

    explode_test_data = df.withColumn(new_column, udfValueToArray(feature))

    return explode_test_data.withColumn(new_column, explode(col=new_column))


def get_element_(v, i):
    return float(v[i])


def main():
    from time import time
    t = time()

    conf = (SparkConf().setMaster('local[4]'))

    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    df = sqlContext.read.option("delimiter", ";").csv("resources/data.csv", header=True,
                                                      inferSchema=True)

    df = df.withColumnRenamed("_c0", "id")

    categorical_features = []

    continuous_features = ['outstanding_balance', 'age', 'dslp', 'dsldc', 'antiquity', 'feature', 'installment_amount',
                           'target']

    target = 'target'

    # GENERATE MULTIPLE VALUES FOR installment_amount
    df = explodeDF(df, "outstanding_balance", "installment_amount")

    # TRANSFORM FEATURES
    stages = transformFeatures(categorical_features, continuous_features, target)

    # FEATURES PIPELINE
    transformed_df = featuresPipeline(stages, df)

    transformed_test_data_rdd = transformed_df.rdd.map(
        lambda x: (x["id"], x["installment_amount"], x["label"], DenseVector(x["features"])))

    dense_data = sqlContext.createDataFrame(transformed_test_data_rdd,
                                            ["id", "installment_amount", "label", "features"])

    # LOAD MODEL
    logistic_model = PipelineModel.load("models/lr-model")

    # TRANSFORM
    logistic_prediction = logistic_model.transform(dense_data)

    print("logistic prediction done\n")

    # RETURN PROBABILITY TARGET = 1.0
    get_element = udf(get_element_, DoubleType())

    df_with_p1 = logistic_prediction.select("id", "installment_amount", "probability") \
        .withColumn("P1", get_element("probability", lit(1))) \
        .select("id", "installment_amount", "P1")

    print("target done\n")

    # RETURN installment_amount WITH HIGHEST P1
    df_with_highest_p1 = df_with_p1.selectExpr(
        "id", "installment_amount", "P1",
        "row_number() over (partition by id order by P1 desc) as tmp"
    ).where(F.col("tmp") == 1).drop("tmp")

    print("highest done\n")

    # WRITE DF AS CSV
    df_with_highest_p1.repartition(1).write.mode("overwrite").option("header", "true").save(
        path='predictions/lr-model', format='csv', mode='append', sep=',')

    elapsed = time() - t

    print(elapsed)


if __name__ == '__main__':
    main()
