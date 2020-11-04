(ns clj-djl.ndarray.shape-manipulation-test
  (:require [clojure.test :refer :all]
            [clj-djl.ndarray :as nd]
            [clj-djl.utils :refer :all]
            [clojure.core.matrix :as matrix])
  (:import [ai.djl.ndarray.types DataType]
           [java.nio FloatBuffer]))

(deftest split-test
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

(deftest flatten-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array (nd/create ndm (float-array [1. 2. 3. 4.]))
          result (nd/create ndm (float-array [1. 2. 3. 4.]))]
      (is (= (nd/flatten array) result)))
    (let [array (nd/create ndm [1. 2. 3. 4.])
          result (nd/create ndm [1. 2. 3. 4.])]
      (is (= (nd/flatten array) result)))
    ;; multi-dim
    (let [array (nd/create ndm (float-array [1. 2. 3. 4.]) (nd/shape [2 2]))
          result (nd/create ndm (float-array [1. 2. 3. 4.]))]
      (is (= (nd/flatten array) result)))
    (let [array (nd/create ndm [1. 2. 3. 4.] [2 2])
          result (nd/create ndm [1. 2. 3. 4.])]
      (is (= (nd/flatten array) result)))
    ;; scalar
    (let [array (nd/create ndm (float 5.))
          result (nd/create ndm (float-array [5.]))]
      (is (= (nd/flatten array ) result)))
    (let [array (nd/create ndm 5.)
          result (nd/create ndm [5.])]
      (is (= (nd/flatten array ) result)))
    ;; zero-dim
    (let [array (nd/create ndm (nd/shape [2 0]))
          result (nd/create ndm (nd/shape [0]))]
      (is (= (nd/flatten array) result)))
    (let [array0 (nd/create ndm (nd/shape [2 0]))
          array1 (nd/create ndm (nd/shape [3 0]))
          array2 (nd/create ndm (nd/shape [0 3]))
          result (nd/create ndm (nd/shape [0]))]
      (is (= (nd/flatten array0)
             (nd/flatten array1)
             (nd/flatten array2)
             result)))))

(deftest reshape-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array (nd/create ndm (float-array [1. 2. 3. 4. 5. 6.]))
          result (-> (nd/create ndm (float-array [1. 2. 3. 4. 5. 6.]))
                     (nd/reshape [2 1 1 3]))]
      (is (= (nd/reshape array [2 1 1 3]) result))
      (is (= (nd/reshape array [-1 1 1 3]) result)))
    (let [array (nd/create ndm [1. 2. 3. 4.])
          result (nd/create ndm [1. 2. 3. 4.])]
      (is (= (nd/flatten array) result)))
    ;; multi-dim
    (let [array (nd/create ndm (float-array [1. 2. 3. 4. 5. 6.]) (nd/shape [3 2]))
          result (nd/create ndm (float-array [1. 2. 3. 4. 5. 6.]) (nd/shape [2 3]))]
      (is (= (nd/reshape array [2 3]) result))
      (is (= (nd/reshape array [2 -1]) result))
      (is (= (nd/reshape array [-1 3]) result)))
    (let [array (nd/create ndm [1. 2. 3. 4. 5. 6.] [3 2])
          result (nd/create ndm [1. 2. 3. 4. 5. 6.] [2 3])]
      (is (= (nd/reshape array [2 3]) result))
      (is (= (nd/reshape array [2 -1]) result))
      (is (= (nd/reshape array [-1 3]) result)))
    ;; scalar
    (let [array (nd/create ndm (float 5.))
          result (nd/create ndm (float-array [5.]))]
      (is (= (nd/reshape array [1]) result)))
    (let [array (nd/create ndm 5.)
          result (nd/create ndm [5.])]
      (is (= (nd/reshape array [1]) result)))
    ;; zero-dim
    (let [array (nd/create ndm (nd/shape [1 0]))
          result (nd/create ndm (nd/shape [2 3 0 1]))]
      (is (= (nd/reshape array [2 3 0 1]) result)))
    (let [array0 (nd/create ndm (nd/shape [2 0]))
          array1 (nd/create ndm (nd/shape [3 0]))
          array2 (nd/create ndm (nd/shape [0 3]))
          result (nd/create ndm (nd/shape [2 3 0 1]))]
      (is (= (nd/reshape array0 [2 3 0 1])
             (nd/reshape array1 [2 3 0 1])
             (nd/reshape array2 [2 3 0 1])
             result)))))





(comment
  (def ndm (nd/new-base-manager))
  (nd/flatten (nd/create ndm (nd/shape [0 2])))
  (nd/flatten (nd/create ndm [0]))
  (def array (nd/reshape (nd/arange ndm 6.) [2 3]))
  array
  (nd/split array [2])
  )
