(ns clj-djl.training.parameter
  (:import [ai.djl.nn Parameter$Type]))

(def beta Parameter$Type/BETA)

(def bias Parameter$Type/BIAS)

(def gamma Parameter$Type/GAMMA)

(def other Parameter$Type/OTHER)

(def running-mean Parameter$Type/RUNNING_MEAN)

(def running-var Parameter$Type/RUNNING_VAR)

(def weight Parameter$Type/WEIGHT)
