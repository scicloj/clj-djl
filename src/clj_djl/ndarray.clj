(ns clj-djl.ndarray
  (:import [ai.djl.ndarray NDManager NDArray NDList NDArrays]
           [ai.djl.ndarray.index NDIndex]
           [ai.djl.ndarray.types Shape DataType])
  (:refer-clojure :exclude [+ - / *
                            = <= < >= >
                            identity to-array
                            min max concat
                            get set]))

(defn new-base-manager []
  (NDManager/newBaseManager))

(def manager (NDManager/newBaseManager))

(defn get-device [ndarray]
  (.getDevice ndarray))

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
  ([manager & col]
   (.zeros manager (shape col)))
  ([manager m & more]
   (zeros manager (into [m] more))))

(defn ones
  ([manager col]
   (.ones manager (shape col)))
  ([manager m & more]
   (ones manager (into [m] more))))

(defn arange
  ([manager stop]
   (.arange manager stop))
  ([manager start stop]
   (.arange manager start stop))
  ([manager start stop step]
   (.arange manager start stop step))
  ([manager start stop step data-type]
   (condp clojure.core/= (type data-type)
     java.lang.String (.arange manager start stop step
                               (DataType/valueOf (.toUpperCase data-type)))
     DataType (.arange manager start stop step data-type)))
  ([manager start stop step data-type device]
   (condp clojure.core/= (type data-type)
     java.lang.String (.arange manager start stop step
                               (DataType/valueOf (.toUpperCase data-type)) device)
     DataType (.arange manager start stop step data-type device))))

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

(defn new-ndlist [array0 array1]
  (new NDList [array0 array1]))

(defn stack [col]
  (NDArrays/stack (NDList. (into-array col))))

(defn random-normal
  ([manager loc scale shape-col data-type]
   (cond
     (instance? java.util.Collection shape-col)
     (.randomNormal manager loc scale (shape shape-col) data-type)
     (instance? ai.djl.ndarray.types.Shape shape-col)
     (.randomNormal manager loc scale shape-col data-type)))
  ([manager loc scale shape-col data-type device]
   (cond
     (instance? java.util.Collection shape-col)
     (.randomNormal manager loc scale (shape shape-col) data-type device)
     (instance? ai.djl.ndarray.types.Shape shape-col)
     (.randomNormal manager loc scale shape-col data-type device)))
  ([manager shape-col]
   (cond
     (instance? java.util.Collection shape-col)
     (.randomNormal manager (shape shape-col))
     (instance? ai.djl.ndarray.types.Shape shape-col)
     (.randomNormal manager shape-col))))

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

(defn dot [array0 array1]
  (.dot array0 array1))

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

(defn get-element
  ([array]
   (get-element array []))
  ([array index]
   (let [index (long-array index)]
     (condp clojure.core/= (.getDataType array)
       DataType/BOOLEAN (.getBoolean array index)
       DataType/INT8 (.getByte array index)
       DataType/INT32 (.getInt array index)
       DataType/INT64 (.getLong array index)
       DataType/FLOAT32 (.getFloat array index)
       DataType/FLOAT64 (.getDouble array index)))))

(defn to-array [array]
  (condp clojure.core/= (.getDataType array)
    DataType/BOOLEAN (.toBooleanArray array)
    DataType/INT8 (.toByteArray array)
    DataType/INT32 (.toIntArray array)
    DataType/INT64 (.toLongArray array)
    DataType/FLOAT32 (.toFloatArray array)
    DataType/FLOAT64 (.toDoubleArray array)))

(defn to-type
  "convert ndarray to data-type, available options are:
  \"int8\" \"uint8\" \"int32\" \"int64\"
  \"float16\" \"float32\" \"float64\"
  \"boolean\" \"string\" \"unknown\""
  [ndarray data-type copy]
  (condp clojure.core/= (type data-type)
    java.lang.String (.toType ndarray (DataType/valueOf (.toUpperCase data-type)) copy)
    DataType (.toType ndarray data-type copy)))

(defn set
  ([array index value]
   (if (vector? index)
     (.set array (NDIndex. (long-array index)) value))))

(defn get
  ([array]
   (.get array (NDIndex. (long-array []))))
  ([array indices & more]
   (cond
     (instance? java.util.Collection indices)
     (.get array (NDIndex. (long-array indices)))
     (instance? java.lang.String indices)
     (.get array (NDIndex. indices (object-array more)))
     :else
     (.get array indices))))

(defn singleton-or-throw [ndlist]
  (.singletonOrThrow ndlist))

(defn head [ndlist]
  (.head ndlist))

(defn attach-gradient
  "Attaches a gradient NDArray to this NDArray and marks it so
  GradientCollector.backward(NDArray) can compute the gradient with respect to it."
  [ndarray]
  (.attachGradient ndarray))
