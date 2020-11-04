(ns clj-djl.training.block-test
  (:require [clojure.test :refer :all]
            [clj-djl.ndarray :as nd]
            [clj-djl.training :as train]
            [clj-djl.model :as model]
            [clj-djl.nn :as nn]
            [clj-djl.training.loss :as loss]
            [clj-djl.utils :refer :all])
  (:import (ai.djl.training.initializer Initializer)))

(deftest flatten-block
  (def config (-> (loss/l2-loss) (train/new-default-training-config) (train/opt-initializer Initializer/ONES)))
  (try-let [model (model/new-instance "model")]
           (model/set-block model (nn/batch-flatten-block))
           (try-let [trainer (model/new-trainer model config)
                     manager (train/get-manager trainer)
                     param-store (train/parameter-store manager false)
                     data (nd/random-uniform manager 0 255 [10 28 28])
                     expected (nd/reshape data [10 (* 28 28)])
                     result (-> model model/get-block (nn/forward param-store (nd/ndlist data) true) nd/head)]
                    (is (= result expected)))))
