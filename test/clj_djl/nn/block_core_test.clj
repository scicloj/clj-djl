(ns clj-djl.nn.block-core-test
  (:require
   [clojure.test :refer [deftest is]]
   [clj-djl.ndarray :as nd]
   [clj-djl.training :as t]
   [clj-djl.model :as m]
   [clj-djl.nn :as nn]
   [clj-djl.training.loss :as loss]
   [clj-djl.training.initializer :as init]
   [clj-djl.utils :refer :all]))

(deftest linear-test
  (let [outsize 3
        input-shape (nd/shape [2 2])]
    (with-open [model (m/model {:name "linear"
                                :block (nn/linear {:units outsize})})
                trainer (t/trainer {:model model
                                    :loss (loss/l2)
                                    :initializer (init/ones)})]
      (t/initialize trainer input-shape)
      (let [ndm  (t/get-manager trainer)
            data (nd/create ndm (float-array [1 2 3 4]) input-shape)
            result (->> data nd/ndlist (t/forward trainer) first)
            expected (-> data
                         (nd/dot (nd/transpose (nd/ones ndm (nd/shape [outsize 2]))))
                         (nd/+ (nd/zeros ndm (nd/shape 2 outsize))))]
        (is (= result expected))))))
