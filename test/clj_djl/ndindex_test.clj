(ns clj-djl.ndindex-test
  (:require [clojure.test :refer :all]
            [clj-djl.ndarray :as nd]
            [clj-djl.utils :refer :all]))

(deftest empty-index-test
  (try-let [ndm (nd/new-base-manager)
            original (nd/create ndm [1. 2. 3. 4.] [2 2])]
           (is (= (nd/get original) original))
           (is (= (nd/get original []) original))))

(deftest fixed-negative-index-test
  (try-let [ndm (nd/new-base-manager)
            original (nd/create ndm [1. 2. 3. 4.] [4])
            expected (nd/create ndm 4.)
            actual (nd/get original "-1")]
           (is (= actual expected))))

(deftest pick-test
  (try-let [ndm (nd/new-base-manager)
            original (nd/create ndm [1. 2. 3. 4.] [2 2])
            expected (nd/create ndm [1. 4.] [2 1])
            actual (nd/get original (-> (nd/new-ndindex)
                                        (.addAllDim)
                                        (.addPickDim (nd/create ndm [0 1]))))]
           (is (= actual expected))))

(deftest get-test
  (try-let [ndm (nd/new-base-manager)
            original (nd/create ndm [1. 2. 3. 4.] [2 2])]
           (is (= (nd/get original (nd/ndindex)) original))

           (let [get-at (nd/get original 0)
                 expected (nd/create ndm [1. 2.])]
             (is (= get-at expected))
             (is (= (nd/get original "0,:") expected))
             (is (= (nd/get original "0,*") expected)))

           (let [get-slice (nd/get original "1:")
                 get-step-slice (nd/get original "1::2")
                 expected (nd/create ndm [3. 4.] [1 2])]
             (is (= get-slice expected))
             (is (= get-step-slice expected)))
           (let [original (-> (nd/arange ndm 120) (nd/reshape [2 3 4 5]))]
             (let [get-ellipsis (nd/get original "0,2, ... ")
                   expected (-> (nd/arange ndm 40 60) (nd/reshape [4 5]))]
               (is (= expected get-ellipsis)))
             (let [get-ellipsis (nd/get original "...,0:2,2") ;; 0:2, include 0, but not 2
                   expected (-> (nd/create ndm [(int 2), 7, 22, 27, 42, 47, 62, 67, 82, 87, 102, 107])
                                (nd/reshape [2 3 2]))]
               (is (= expected get-ellipsis)))
             (let [get-ellipsis (nd/get original "1,...,2,3:5:2")
                   expected (-> (nd/create ndm [(int 73) 93 113])
                                (nd/reshape [3 1]))]
               (is (= expected get-ellipsis)))
             (let [get-ellipsis (nd/get original "...")]
               (is (= get-ellipsis original))))
           (let [original (-> (nd/arange ndm 10) (nd/reshape [2 5]))
                 bool (nd/create ndm [true false])
                 expected (-> (nd/arange ndm 5) (nd/reshape [1 5]))]
             (is (= (.get original bool) expected)))))

(deftest set-array
  (try-let [ndm (nd/new-base-manager)]
           (let [original (nd/create ndm (float-array [1 2 3 4]) (nd/new-shape [2 2]))
                 expected (nd/create ndm (float-array [9 10 3 4]) (nd/new-shape [2 2]))
                 value (nd/create ndm (float-array [9 10]))]
             (nd/set original [0] value)
             (is (= original expected)))
           (let [original (nd/create ndm [1 2 3 4] [2 2])
                 expected (nd/create ndm [9 10 3 4] [2 2])
                 value (nd/create ndm [9 10])]
             (nd/set original [0] value)
             (is (= original expected)))
           (let [original (-> (nd/arange ndm 0 8) (nd/reshape [2 4]))
                 expected (nd/create ndm [(int 0) 1 9 10 4 5 11 12] [2 4])]
             (nd/set original ":, 2:" (-> (nd/arange ndm 9 13) (nd/reshape [2 2])))
             (is (= original expected)))))
