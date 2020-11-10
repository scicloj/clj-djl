(ns clj-djl.training.gradient-collector-test
  (:require [clojure.test :refer :all]
            [clj-djl.model :as m]
            [clj-djl.ndarray :as nd]
            [clj-djl.training :as t]
            [clj-djl.nn :as nn]
            [clj-djl.training.loss :as l])
  (:import [ai.djl.training.initializer Initializer]))

(deftest autograd-test
  (with-open [model (m/new-instance "model")
              ndm (nd/new-base-manager)]
    (m/set-block model (nn/identity-block))
    (with-open [trainer (t/new-trainer model
                                       (t/training-config
                                        {:loss (l/l2-loss)
                                         :initializer Initializer/ONES}))
                gc (t/new-gradient-collector trainer)]
      (let [lhs (nd/create ndm (float-array [6 -9 -12 15 0 4]) [2 3])
            rhs (nd/create ndm (float-array [2 3 -4]) [3 1])
            expected (nd/create ndm (float-array [24 -54 96 60 0 -32]) [2 3])]
        (t/attach-gradient lhs)
        (let [result (nd/dot (nd/* lhs lhs) rhs)]
          (t/backward gc result)
          (let [grad (t/get-gradient lhs)]
            (is (= expected grad))
            (println grad)
            (.close grad)
            (is (= expected (t/get-gradient lhs)))))))))
