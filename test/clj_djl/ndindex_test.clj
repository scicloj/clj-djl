(ns clj-djl.ndindex-test
  (:require [clojure.test :refer :all]
            [clj-djl.ndarray :as nd]
            [clj-djl.utils :refer :all]))

(deftest empty-index
  (try-let [ndm (nd/new-base-manager)
            original (nd/create ndm [1. 2. 3. 4.] [2 2])]
           (is (= (nd/get original) original))
           (is (= (nd/get original []) original))))

(deftest fixed-negative-index
  (try-let [ndm (nd/new-base-manager)
            original (nd/create ndm [1. 2. 3. 4.] [4])
            expected (nd/create ndm 4.)
            actual (nd/get original "-1")]
           (is (= actual expected))))

(deftest pick
  (try-let [ndm (nd/new-base-manager)
            original (nd/create ndm [1. 2. 3. 4.] [2 2])
            expected (nd/create ndm [1. 4.] [2 1])
            actual (nd/get original (-> (nd/new-ndindex)
                                        (.addAllDim)
                                        (.addPickDim (nd/create ndm [0 1]))))]
           (is (= actual expected))))

(deftest get
  (try-let [ndm (nd/new-base-manager)
            original (nd/create ndm [1. 2. 3. 4.] [2 2])]
           (is (= (nd/get original (nd/ndindex)) original))))


(deftest set-array
  (testing "set array"
    (def ndm (nd/new-base-manager))
    (def original (nd/create ndm (float-array [1 2 3 4]) (nd/new-shape [2 2])))
    (def expected (nd/create ndm (float-array [9 10 3 4]) (nd/new-shape [2 2])))
    (def value (nd/create ndm (float-array [9 10])))
    (nd/set original [0] value)
    (is (= original expected)))
  (testing "string index"
    ;;(def original (-> (nd/arange ndm 0 8) (nd/reshape [2 4])))
    ;;(def expected (nd/create ndm (int-array [0 1 9 10 4 5 11 12]) (nd/new-shape [2 4])))
    ;;(nd/set original ":, 2:" (-> (nd/arange ndm 9 13) (nd/reshape [2 2])))
    ;;(is (= original expected))
    ))
