(ns clj-djl.ndarray.ndarray-shape-manipulation-test
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

(deftest expand-dim-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array (nd/create ndm (float-array [1. 2.]))
          expected (nd/create ndm (float-array [1. 2.]) (nd/shape [1 2]))]
      (is (= (nd/expand-dims array 0) expected)))
    (let [array (nd/create ndm [1. 2.])
          expected (nd/create ndm [1. 2.] [1 2])]
      (is (= (nd/expand-dims array 0) expected)))

    ;; multi-dim
    (let [array (nd/create ndm (float-array [1. 2. 3. 4.]) (nd/shape [2 2]))
          expected (nd/create ndm (float-array [1. 2. 3. 4.]) (nd/shape [2 1 2]))]
      (is (= (nd/expand-dims array 1) expected)))
    (let [array (nd/create ndm [1. 2. 3. 4.] [2 2])
          expected (nd/create ndm [1. 2. 3. 4.] [2 1 2])]
      (is (= (nd/expand-dims array 1) expected)))

    ;; scalar
    (let [array (nd/create ndm (float 4.))
          expected (nd/create ndm (float-array [4.]))]
      (is (= (nd/expand-dims array 0) expected)))
    (let [array (nd/create ndm 4.)
          expected (nd/create ndm [4.])]
      (is (= (nd/expand-dims array 0) expected)))

    ;; zero-dim
    (let [array (nd/create ndm (nd/shape [2 1 0]))
          expected (nd/create ndm (nd/shape [2 1 1 0]))]
      (is (= (nd/expand-dims array 2) expected)))))

(deftest squeeze-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array (nd/ones ndm (nd/shape [1 2 1 3 1]))
          expected (nd/ones ndm (nd/shape [2 3]))]
      (is (= (nd/squeeze array) expected)))
    (let [array (nd/ones ndm [1 2 1 3 1])
          expected (nd/ones ndm [2 3])]
      (is (= (nd/squeeze array) expected)))

    ;; scalar
    (let [array (nd/create ndm (float 2.))]
      (is (= (nd/squeeze array) array))
      (is (= (nd/squeeze array 0) array))
      (is (= (nd/squeeze array (int-array [0])) array))
      (is (= (nd/squeeze array [0]) array)))

    ;; zero-dim
    (let [array (nd/create ndm (nd/shape [1 0 1 3 1]))]
      (let [expected (nd/create ndm (nd/shape [0 3]))]
        (is (= (nd/squeeze array) expected)))
      (let [expected (nd/create ndm (nd/shape [1 0 3 1]))]
        (is (= (nd/squeeze array 2) expected)))
      (let [expected (nd/create ndm (nd/shape [0 1 3]))]
        (is (= (nd/squeeze array (int-array [0 4])) expected))
        (is (= (nd/squeeze array [0 4]) expected))))))

(deftest stack-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array1 (nd/create ndm (float-array [1. 2.]))
          array2 (nd/create ndm (float-array [3. 4.]))]
      (let [expected (nd/create ndm (float-array [1. 2. 3. 4.]) (nd/shape 2 2))]
        (is (= (nd/stack array1 array2) expected))
        (is (= (nd/stack (nd/ndlist array1 array2)) expected))
        (is (= (nd/stack [array1 array2]) expected)))
      (let [expected (nd/create ndm (float-array [1. 3. 2. 4.]) (nd/shape 2 2))]
        (is (= (nd/stack array1 array2 1) expected))
        (is (= (nd/stack (nd/ndlist array1 array2) 1) expected))
        (is (= (nd/stack [array1 array2] 1) expected))))
    (let [array1 (nd/create ndm (float-array [1. 2. 3. 4.]) (nd/shape 2 2))
          array2 (nd/create ndm (float-array [5. 6. 7. 8.]) (nd/shape 2 2))]
      (let [expected (nd/create ndm (float-array [1. 2. 3. 4. 5. 6. 7. 8.]) (nd/shape 2 2 2))]
        (is (= (nd/stack array1 array2) expected))
        (is (= (nd/stack (nd/ndlist array1 array2)) expected))
        (is (= (nd/stack [array1 array2]) expected)))
      (let [expected (nd/create ndm (float-array [1. 2. 5. 6. 3. 4. 7. 8.]) (nd/shape 2 2 2))]
        (is (= (nd/stack array1 array2 1) expected))
        (is (= (nd/stack (nd/ndlist array1 array2) 1) expected))
        (is (= (nd/stack [array1 array2] 1) expected))))

    ;; scalar
    (let [array1 (nd/create ndm (float 5.))
          array2 (nd/create ndm (float 4.))]
      (let [expected (nd/create ndm (float-array [5. 4.]))]
        (is (= (nd/stack array1 array2) expected))
        (is (= (nd/stack (nd/ndlist array1 array2)) expected))
        (is (= (nd/stack [array1 array2]) expected))))

    ;;zero-dim
    (let [array1 (nd/create ndm (nd/shape 0 0))]
      (let [expected (nd/create ndm (nd/shape 2 0 0))]
        (is (= (nd/stack array1 array1) expected))
        (is (= (nd/stack (nd/ndlist array1 array1)) expected))
        (is (= (nd/stack [array1 array1]) expected)))
      (let [expected (nd/create ndm (nd/shape 0 2 0))]
        (is (= (nd/stack array1 array1 1) expected))
        (is (= (nd/stack (nd/ndlist array1 array1) 1) expected))
        (is (= (nd/stack [array1 array1] 1) expected)))
      (let [expected (nd/create ndm (nd/shape 0 0 2))]
        (is (= (nd/stack array1 array1 2) expected))
        (is (= (nd/stack (nd/ndlist array1 array1) 2) expected))
        (is (= (nd/stack [array1 array1] 2) expected))))))

(deftest concat-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array1 (nd/create ndm (float-array [1.]))
          array2 (nd/create ndm (float-array [2.]))
          expected (nd/create ndm (float-array [1. 2.]))]
      (is (= (nd/concat (nd/ndlist array1 array2) 0) expected))
      (is (= (nd/concat [array1 array2] 0) expected))
      (is (= (nd/concat array1 array2) expected)))
    (let [array1 (nd/create ndm [1.])
          array2 (nd/create ndm [2.])
          expected (nd/create ndm [1. 2.])]
      (is (= (nd/concat (nd/ndlist array1 array2) 0) expected))
      (is (= (nd/concat [array1 array2] 0) expected))
      (is (= (nd/concat array1 array2) expected)))

    (let [array1 (nd/create ndm (float-array [1. 2. 3. 4.]) (nd/shape [2 2]))
          array2 (nd/create ndm (float-array [5. 6. 7. 8.]) (nd/shape [2 2]))
          expected (nd/create ndm (float-array [1. 2. 3. 4. 5. 6. 7. 8.]) (nd/shape [4 2]))]
      (is (= (nd/concat (nd/ndlist array1 array2)) expected))
      (is (= (nd/concat [array1 array2]) expected))
      (is (= (nd/concat array1 array2) expected)))
    (let [array1 (nd/create ndm [1. 2. 3. 4.] (nd/shape [2 2]))
          array2 (nd/create ndm [5. 6. 7. 8.] (nd/shape [2 2]))
          expected (nd/create ndm [1. 2. 3. 4. 5. 6. 7. 8.] (nd/shape [4 2]))]
      (is (= (nd/concat (nd/ndlist array1 array2)) expected))
      (is (= (nd/concat [array1 array2]) expected))
      (is (= (nd/concat array1 array2) expected)))

    ;; zero-dim
    (let [array1 (nd/create ndm (nd/shape [0 1]))]
      (let [expected (nd/create ndm (nd/shape [0 1]))]
        (is (= (nd/concat array1 array1) expected))
        (is (= (nd/concat (nd/ndlist array1 array1)) expected))
        (is (= (nd/concat [array1 array1]) expected)))
      (let [expected (nd/create ndm (nd/shape [0 2]))]
        (is (= (nd/concat array1 array1 1) expected))
        (is (= (nd/concat (nd/ndlist array1 array1) 1) expected))
        (is (= (nd/concat [array1 array1] 1) expected))))))

(deftest concat-ndlist-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array1 (nd/create ndm (float 1.))
          array2 (nd/create ndm (float 2.))]
      (is (thrown? java.lang.IllegalArgumentException (nd/concat (nd/ndlist array1 array2))))
      (is (thrown? java.lang.IllegalArgumentException (nd/concat [array1 array2]))))
    (let [array1 (nd/create ndm 1.)
          array2 (nd/create ndm 2.)]
      (is (thrown? java.lang.IllegalArgumentException (nd/concat (nd/ndlist array1 array2))))
      (is (thrown? java.lang.IllegalArgumentException (nd/concat [array1 array2]))))))

(comment
  (def ndm (nd/new-base-manager))
  (nd/flatten (nd/create ndm (nd/shape [0 2])))
  (nd/flatten (nd/create ndm [0]))
  (def array (nd/reshape (nd/arange ndm 6.) [2 3]))
  array
  (nd/split array [2])
  )
