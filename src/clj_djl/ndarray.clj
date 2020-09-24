(ns clj-djl.ndarray
  (:import [ai.djl.ndarray NDArray NDManager]
           [ai.djl.ndarray.types Shape])
  (:refer-clojure :exclude [+ - / *
                            = <= < >= >
                            identity
                            min max concat]))

(def manager (NDManager/newBaseManager))

(defn shape
  ([col]
   (Shape. col))
  ([m n]
   (shape [m n])))

(defn get-shape [ndarray]
  (.getShape ndarray))

(defn reshape [ndarray new-shape]
  (.reshape ndarray (long-array new-shape)))

(defn size
  "calc the seize of a ndarray."
  [ndarray]
  (.size ndarray))

(defn scalar? [ndarray]
  (.isScalar ndarray))

(defn zeros
  ([col]
   (.zeros manager (shape col)))
  ([m & more]
   (zeros (into [m] more))))

(defn ones
  ([col]
   (.ones manager (shape col)))
  ([m & more]
   (ones (into [m] more))))

(defn arange [start end]
  (.arange manager start end))

(defn create
  ([shape-col]
   (.create manager (shape shape-col)))
  ([col shape-col]
   (.create manager (float-array col) (shape shape-col))))

(defn random-normal
  ([loc scale shape-col data-type]
   (.randomNormal manager loc scale (shape shape-col) data-type))
  ([shape-col]
   (.randomNormal manager (shape shape-col))))

(defn + [array0 array1]
  (.add array0 array1))

(defn - [array0 array1]
  (.sub array0 array1))

(defn * [array0 array1]
  (.mul array0 array1))

(defn / [array0 array1]
  (.div array0 array1))

(defn ** [array0 array1]
  (.pow array0 array1))

(defn = [array0 array1]
  (.eq array0 array1))

(defn exp [array0]
  (.exp array0))

(defn sum [array0]
  (.sum array0))

(defn concat
  ([array0 array1]
   (concat array0 array1 :axis 0))
  ([array0 array1 & {axis :axis}]
   (.concat array0 array1 axis)))
