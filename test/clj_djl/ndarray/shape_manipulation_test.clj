(ns clj-djl.ndarray.shape-manipulation-test
  (:require [clojure.test :refer :all]
            [clj-djl.ndarray :as nd]
            [clj-djl.utils :refer :all]
            [clojure.core.matrix :as matrix])
  (:import [ai.djl.ndarray.types DataType]
           [java.nio FloatBuffer]))

(deftest shape-manipulation-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array (nd/arange ndm 18.)
          result (nd/split array 18)]
      (is (= (nd/get result 0) (nd/create ndm (float-array [0.]))))
      (is (= (nd/get result 8) (nd/create ndm (float-array [8.]))))
      (is (= (nd/get result 17) (nd/create ndm (float-array [17.])))))
    (let [array (nd/create ndm [1. 2. 3. 4.])]
      (let [result (nd/split array 2)]
        (is (= (nd/get result 0) (nd/create ndm [1. 2.])))
        (is (= (nd/get result 1) (nd/create ndm [3. 4.]))))
      (let [result (nd/split array (long-array [2]))]
        (is (= (nd/get result 0) (nd/create ndm [1. 2.])))
        (is (= (nd/get result 1) (nd/create ndm [3. 4.]))))
      (let [array (-> (nd/arange ndm 6.) (nd/reshape [2 3]))
            result (nd/split array (long-array [0]))]
        (is (= (nd/singleton-or-throw result) array)))
      (let [result (nd/split array [2])]
        (is (= (nd/get result 0) (nd/create ndm [1. 2.])))
        (is (= (nd/get result 1) (nd/create ndm [3. 4.])))))
    (let [array (-> (nd/arange ndm 6.) (nd/reshape [2 3]))
          result (nd/split array [0])]
      (is (= (nd/singleton-or-throw result) array)))))

(comment
  (def ndm (nd/new-base-manager))
  (def array (nd/reshape (nd/arange ndm 6.) [2 3]))
  array
  (nd/split array [2])

  )
