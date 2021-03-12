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

(deftest add-ndarray-test
  (with-open [ndm (nd/new-base-manager)]
    (let [addend (nd/create ndm (float-array [1 2 3 4]))
          addendum (nd/create ndm (float-array [2 3 4 5]))
          result (nd/+ addend addendum)
          expected (nd/create ndm (float-array [3 5 7 9]))]
      (is (not= result addend))
      (is (= result expected)))
    (let [addend (nd/create ndm (float-array [1 2 3 4]))
          addendum (nd/create ndm (float-array [2 3 4 5]))
          result (nd/+! addend addendum)
          expected (nd/create ndm (float-array [3 5 7 9]))]
      (is (= result expected addend)))
    (let [to-add-all [(nd/create ndm (float-array [1 2 3 4]) [2 2])
                      (nd/create ndm (float-array [4 3 2 1]) [2 2])
                      (nd/create ndm (float-array [2 2 2 2]) [2 2])]
          to-add-all-array (into-array ai.djl.ndarray.NDArray to-add-all)
          expected (nd/create ndm (float-array [7 7 7 7]) [2 2])
          add-all (ai.djl.ndarray.NDArrays/add
                   to-add-all-array)]
      (is (= expected add-all))
      (is (not= expected (aget to-add-all-array 0))))
    (let [to-add-all [(nd/create ndm (float-array [1 2 3 4]) [2 2])
                      (nd/create ndm (float-array [4 3 2 1]) [2 2])
                      (nd/create ndm (float-array [2 2 2 2]) [2 2])]
          to-add-all-array (into-array ai.djl.ndarray.NDArray to-add-all)
          expected (nd/create ndm (float-array [7 7 7 7]) [2 2])
          add-all (ai.djl.ndarray.NDArrays/addi
                   to-add-all-array)]
      (is (= expected add-all (aget to-add-all-array 0))))))

(deftest sub-scalar-test
  (with-open [ndm (nd/new-base-manager)]
    (let [minuend (nd/create ndm (float-array [6 9 12 11 0]))
          result (ai.djl.ndarray.NDArrays/sub minuend 3)
          in-place-result (ai.djl.ndarray.NDArrays/subi minuend 3)
          expected (nd/create ndm (float-array [3 6 9 8 -3]))]
      (is (= result expected in-place-result)))
    (let [minuend (nd/create ndm (float-array [6 9 12 11 0]))
          result (nd/- minuend 3)
          in-place-result (nd/-! minuend 3)
          expected (nd/create ndm (float-array [3 6 9 8 -3]))]
      (is (= result expected in-place-result)))))

(deftest sub-ndarray-test
  (with-open [ndm (nd/new-base-manager)]
    (let [minuend (nd/create ndm (float-array [6 9 12 15 0]))
          subtrahend (nd/create ndm (float-array [2 3 4 5 6]))
          result (ai.djl.ndarray.NDArrays/sub minuend subtrahend)
          in-place-result (ai.djl.ndarray.NDArrays/subi minuend subtrahend)
          expected (nd/create ndm (float-array [4 6 8 10 -6]))]
      (is (= result expected in-place-result)))
    (let [minuend (nd/create ndm (float-array [6 9 12 15 0]))
          subtrahend (nd/create ndm (float-array [2 3 4 5 6]))
          result (nd/- minuend subtrahend)
          in-place-result (nd/-! minuend subtrahend)
          expected (nd/create ndm (float-array [4 6 8 10 -6]))]
      (is (= result expected in-place-result)))))


(deftest reverse-sub-scalar-test
  (with-open [ndm (nd/new-base-manager)]
    (let [subtrahend (nd/create ndm (float-array [6 91 12 215 180]))
          result (ai.djl.ndarray.NDArrays/sub 180 subtrahend)
          in-place-result (ai.djl.ndarray.NDArrays/subi 180 subtrahend)
          expected (nd/create ndm (float-array [174 89 168 -35 0]))]
      (is (= result expected in-place-result)))
    ;; let - accept both ndarray and scaler as the first parameter?
    #_(let [subtrahend (nd/create ndm (float-array [6 91 12 215 180]))
          result (nd/- 180 subtrahend)
          in-place-result (nd/-! 180 subtrahend)
          expected (nd/create ndm (float-array [174 89 168 -35 0]))]
      (is (= result expected in-place-result)))))
