from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

from pyspark.ml import Pipeline

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

from pyspark.ml.linalg import DenseVector

from pyspark.ml.classification import LogisticRegression


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

    # TRANSFORM FEATURES
    stages = transformFeatures(categorical_features, continuous_features, target)

    # FEATURES PIPELINE
    transformed_train_data = featuresPipeline(stages, df)

    transformed_train_data_rdd = transformed_train_data.rdd.map(lambda x: (x["label"], DenseVector(x["features"])))
    dense_train_data = sqlContext.createDataFrame(transformed_train_data_rdd, ["label", "features"])

    lr = LogisticRegression(labelCol="label",
                            featuresCol="features",
                            maxIter=10,
                            regParam=0.3)

    # FIT
    pipeline = Pipeline(stages=[lr])

    logistic_model = pipeline.fit(dense_train_data)

    # SAVE MODEL
    logistic_model.write().overwrite().save("models/lr-model")

    print("logistic model done\n")

    elapsed = time() - t
    print(elapsed)


if __name__ == '__main__':
    main()
