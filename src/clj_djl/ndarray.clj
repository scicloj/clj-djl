(ns clj-djl.ndarray
  (:import [ai.djl.ndarray NDArray NDManager]
           [ai.djl.ndarray.types Shape DataType])
  (:refer-clojure :exclude [+ - / *
                            = <= < >= >
                            identity to-array
                            min max concat]))

(defn new-base-manager []
  (NDManager/newBaseManager))

(def manager (NDManager/newBaseManager))

(defn shape
  ([]
   (shape []))
  ([col]
   (Shape. col))
  ([m n]
   (shape [m n])))

(def new-shape shape)

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

(defn arange
  ([stop]
   (.arange stop))
  ([start stop]
   (.arange manager start stop))
  ([start stop step]
   (.arange manager start stop step))
  ([start stop step data-type]
   (.arange manager start stop step data-type))
  ([start stop step data-type device]
   (.arange manager start stop step data-type device)))

(defn create
  ([manager data]
   (.create manager data))
  ([manager data shape]
   (.create manager data shape))
  #_([manager shape]
   (.create shape))
  #_([manager shape data-type]
   (.create shape data-type))
  #_([manager col shape-col]
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

(defn get-element [array index]
  (let [index (long-array index)]
    (condp clojure.core/= (.getDataType array)
      DataType/BOOLEAN (.getBoolean array index)
      DataType/INT8 (.getByte array index)
      DataType/INT32 (.getInt array index)
      DataType/INT64 (.getLong array index)
      DataType/FLOAT32 (.getFloat array index)
      DataType/FLOAT64 (.getDouble array index))))

(defn to-array [array]
  (condp clojure.core/= (.getDataType array)
    DataType/BOOLEAN (.toBooleanArray array)
    DataType/INT8 (.toByteArray array)
    DataType/INT32 (.toIntArray array)
    DataType/INT64 (.toLongArray array)
    DataType/FLOAT32 (.toFloatArray array)
    DataType/FLOAT64 (.toDoubleArray array)))
