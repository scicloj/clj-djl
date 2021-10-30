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
          _ (nd/set-requires-gradient lhs true)
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


(deftest abs-test
  (with-open [ndm (nd/base-manager)]
    (let [array (nd/create ndm [0 -1 1 -2 2])
          result (nd/abs array)
          expected (nd/create ndm [0 1 1 2 2])]
      (is (= result expected)))
    (let [array (nd/create ndm (float-array [0 -1 1 -2.2 2.2]))
          result (nd/abs array)
          expected (nd/create ndm (float-array [0 1 1 2.2 2.2]))]
      (is (= result expected)))))

(deftest acos-test
  (with-open [ndm (nd/base-manager)]
  (let [array (nd/create ndm [1.0 -1.0])
        result (nd/acos array)
        expected (nd/create ndm [0.0  Math/PI])]
    (is (= result expected)))))

(deftest transpose-test
  (with-open [ndm (nd/base-manager)]
    (let [array (nd/create ndm [1])
          result (nd/transpose array)
          expected (nd/create ndm [1])]
      (is (= result expected))))
  (with-open [ndm (nd/base-manager)]
    (let [array (nd/create ndm [1 2 3])
          result (nd/transpose array)
          expected (nd/create ndm [1 2 3])]
      (is (= result expected))))
  (with-open [ndm (nd/base-manager)]
    (let [array (nd/create ndm [1 2 3] [1 3])
          result (nd/transpose array)
          expected (nd/create ndm [1 2 3] [3 1])]
      (is (= result expected))))
  (with-open [ndm (nd/base-manager)]
    (let [array (nd/create ndm [1 2 3 4 5 6] [2 3])
          result (nd/transpose array)
          expected (nd/create ndm [1 4 2 5 3 6] [3 2])]
      (is (= result expected)))))

(deftest argmax-test
  (with-open [ndm (nd/base-manager)]
    (let [array (nd/create ndm [1 2 3 4 5 6] [2 3])
          result (nd/argmax array)
          expected (nd/create ndm 5)]
      (is (= result expected)))
    (let [array (nd/create ndm [1 2 3 4 5 6] [2 3])
          result (nd/argmax array 0)
          expected (nd/create ndm [1 1 1])]
      (is (= result expected)))
    (let [array (nd/create ndm [1 2 3 4 5 6] [2 3])
          result (nd/argmax array 1)
          expected (nd/create ndm [2 2])]
      (is (= result expected)))
    (let [array (nd/create ndm [1 2 3
                                4 5 6
                                7 8 9
                                10 11 12] [2 2 3])
          result (nd/argmax array 0)
          expected (nd/create ndm [1 1 1 1 1 1] [2 3])]
      (is (= result expected)))
    (let [array (nd/create ndm [1 2 3
                                4 5 6
                                7 8 9
                                10 11 12] [2 2 3])
          result (nd/argmax array 1)
          expected (nd/create ndm [1 1 1 1 1 1] [2 3])]
      (is (= result expected)))
    (let [array (nd/create ndm [1 2 3
                                4 5 6
                                7 8 9
                                10 11 12] [2 2 3])
          result (nd/argmax array 2)
          expected (nd/create ndm [2 2 2 2] [2 2])]
      (is (= result expected)))))

(deftest argmin-test
  (with-open [ndm (nd/base-manager)]
    (let [array (nd/create ndm [1 2 3 4 5 6] [2 3])
          result (nd/argmin array)
          expected (nd/create ndm 0)]
      (is (= result expected)))
    (let [array (nd/create ndm [1 2 3 4 5 6] [2 3])
          result (nd/argmin array 0)
          expected (nd/create ndm [0 0 0])]
      (is (= result expected)))
    (let [array (nd/create ndm [1 2 3 4 5 6] [2 3])
          result (nd/argmin array 1)
          expected (nd/create ndm [0 0])]
      (is (= result expected)))
    (let [array (nd/create ndm [1 2 3
                                4 5 6
                                7 8 9
                                10 11 12] [2 2 3])
          result (nd/argmin array 0)
          expected (nd/create ndm [0 0 0 0 0 0] [2 3])]
      (is (= result expected)))
    (let [array (nd/create ndm [1 2 3
                                4 5 6
                                7 8 9
                                10 11 12] [2 2 3])
          result (nd/argmin array 1)
          expected (nd/create ndm [0 0 0 0 0 0] [2 3])]
      (is (= result expected)))
    (let [array (nd/create ndm [1 2 3
                                4 5 6
                                7 8 9
                                10 11 12] [2 2 3])
          result (nd/argmin array 2)
          expected (nd/create ndm [0 0 0 0] [2 2])]
      (is (= result expected)))))

(deftest argsort-test
  (with-open [ndm (nd/base-manager)]
    (let [array (nd/create ndm [0 1 2 3] [2 2])
          result (nd/argsort array 0)
          expected (nd/create ndm [0 0 1 1] [2 2])]
      (is (= result expected)))
    (let [array (nd/create ndm [0 1 2 3] [2 2])
          result (nd/argsort array 0 false)
          expected (nd/create ndm [1 1 0 0] [2 2])]
      (is (= result expected)))
    (let [array (nd/create ndm [12 11 10 9 8 7 6 5 4 3 2 1] [2 2 3])
          result (nd/argsort array 0)
          expected (nd/create ndm [1 1 1 1 1 1 0 0 0 0 0 0] [2 2 3])]
      (is (= result expected)))
    (let [array (nd/create ndm [12 11 10 9 8 7 6 5 4 3 2 1] [2 2 3])
          result (nd/argsort array 0 false)
          expected (nd/create ndm [0 0 0 0 0 0 1 1 1 1 1 1] [2 2 3])]
      (is (= result expected)))
    (let [array (nd/create ndm [12 11 10
                                9 8 7
                                6 5 4
                                3 2 1] [2 2 3])
          result (nd/argsort array 1)
          expected (nd/create ndm [1 1 1 0 0 0 1 1 1 0 0 0] [2 2 3])]
      (is (= result expected)))))

(deftest sort-test
  (with-open [ndm (nd/base-manager)]
    (let [array (nd/create ndm [1 3 4 2] [2 2])
          result (nd/sort array)
          expected (nd/create ndm [1 3 2 4] [2 2])]
      (is (= result expected)))
    (let [array (nd/create ndm [1 4 3 1] [2 2])
          result (.sort array 0)
          expected (nd/create ndm [1 1 3 4] [2 2])]
      (is (= result expected)))
    (let [array (nd/create ndm [1 4 3 1] [2 2])
          result (nd/sort array 1)
          expected (nd/create ndm [1 4 1 3] [2 2])]
      (is (= result expected)))
    (let [array (nd/create ndm [0. 2. 4. 6. 7. 5. 3. 1.] [2 2 2])
          result (nd/sort array 0)
          expected (nd/create ndm [0. 2. 3. 1. 7. 5. 4. 6.] [2 2 2])]
      (is (= result expected)))))
