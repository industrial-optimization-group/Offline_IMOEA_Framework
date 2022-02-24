library(flacco)

## (1) Create some example-data
X = createInitialSample(n.obs = 500, dim = 2)
f = function(x) sum(sin(x) * x^2 + (x - 0.5)^3)
y = apply(X, 1, f)

## (2) Compute the feature object
feat.object = createFeatureObject(X = X, y = y)

## (3) Have a look at feat.object
print(feat.object)

## (4) Check, which feature sets are available
listAvailableFeatureSets()

## (5) Calculate a specific feature set, e.g. the ELA meta model
featureSet = calculateFeatureSet(feat.object, set = "ela_meta")