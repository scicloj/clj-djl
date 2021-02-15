(ns clj-djl.ndarray.ndarray-element-arithmetic-test
  (:require
   [clojure.test :refer [deftest is]]
   [clj-djl.ndarray :as nd]
   [clj-djl.model :as m]
   [clj-djl.training :as t]
   [clj-djl.nn :as nn]
   [clj-djl.training.loss :as loss]
   [clj-djl.training.initializer :as init]))

(deftest add-scalar-test
  (with-open [model    (m/model {:name "model"
                                 :block (nn/identity-block)})
              manager  (m/get-ndmanager model)
              trainer  (t/trainer {:model model
                                   :loss (loss/l2)
                                   :initializer (init/ones)})
              grad-col (t/gradient-collector trainer)]
    (let [lhs (nd/create manager (float-array [1 2 3 4]))
          _ (nd/attach-gradient lhs)
          result (nd/+ lhs 2)
          _ (t/backward grad-col result)
          expected (nd/create manager (float-array [3 4 5 6]))
          expected-gradient (nd/create manager (float-array [1 1 1 1]))
          result-gradient (t/get-gradient lhs)]
      (is (not= result lhs))
      (is (= result expected))
      (is (= result-gradient expected-gradient)))
    ;; inplace add
    (let [lhs (nd/create manager (float-array [1 2 3 4]))
          result (nd/+! lhs 2)
          expected (nd/create manager (float-array [3 4 5 6]))]
      (is (= result expected lhs )))))
