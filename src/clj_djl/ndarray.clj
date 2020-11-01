(ns clj-djl.ndarray
  (:require [clojure.core.matrix :as matrix])
  (:import [ai.djl.ndarray NDManager NDArray NDList NDArrays]
           [ai.djl.ndarray.index NDIndex]
           [ai.djl.ndarray.types Shape DataType]
           [ai.djl Device]
           [java.nio ByteBuffer IntBuffer LongBuffer FloatBuffer DoubleBuffer])
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

(defn default-device []
  (Device/defaultDevice))

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
  (cond
    (instance? java.util.Collection new-shape)
    (.reshape ndarray (long-array new-shape))
    (instance? ai.djl.ndarray.types.Shape new-shape)
    (.reshape ndarray new-shape)))

(defn size
  "calc the seize of a ndarray."
  [ndarray]
  (.size ndarray))

(defn scalar? [ndarray]
  (.isScalar ndarray))

(def datatype-map {:int8 DataType/INT8
                   :int32 DataType/INT32
                   :int64 DataType/INT64
                   :float16 DataType/FLOAT16
                   :float32 DataType/FLOAT32
                   :float64 DataType/FLOAT64})

(defn zeros
  ([manager shape]
   (.zeros manager (new-shape shape)))
  ([manager shape data-type]
   (let [local-data-type (if (keyword? data-type) (datatype-map data-type) data-type)
         local-shape (if (sequential? shape) (new-shape shape) shape)]
     (.zeros manager local-shape local-data-type)))
  ([manager shape data-type device]
   (let [local-data-type (if (keyword? data-type) (datatype-map data-type) data-type)
         local-shape (if (sequential? shape) (new-shape shape) shape)]
     (.zeros manager local-shape local-data-type)) device))

(defn ones
  ([manager shape]
   (.ones manager (new-shape shape)))
  ([manager shape data-type]
   (let [local-data-type (if (keyword? data-type) (datatype-map data-type) data-type)
         local-shape (if (sequential? shape) (new-shape shape) shape)]
     (.ones manager local-shape local-data-type)))
  ([manager shape data-type device]
   (let [local-data-type (if (keyword? data-type) (datatype-map data-type) data-type)
         local-shape (if (sequential? shape) (new-shape shape) shape)]
     (.ones manager local-shape local-data-type)) device))

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


(defmulti create
  (fn [manager data & more]
    (sequential? data)))

(defmethod create :default
  ([manager data]
   (.create manager data))
  ([manager data1 data2]
   (let [param2 (if (sequential? data2)
                  (shape data2)
                  data2)]
     (.create manager data1 param2)))
  ([manager data1 data2 data3]
   (.create manager data1 data2 data3)))

(defmethod create true
  ([manager data]
   (let [param-shape (new-shape (long-array (matrix/shape data)))]
     (create manager data param-shape)))
  ([manager param1 param2]
   (let [flat (flatten param1)
         param-shape (if (sequential? param2)
                       (shape param2)
                       param2)]
     (condp clojure.core/= (type (first flat))
       java.lang.Boolean (.create manager (boolean-array flat) param-shape)
       java.lang.Byte (.create manager (byte-array flat) param-shape)
       java.lang.Integer (.create manager (int-array flat) param-shape)
       java.lang.Long (.create manager (long-array flat) param-shape)
       java.lang.Float (.create manager (float-array flat) param-shape)
       java.lang.Double (.create manager (double-array flat) param-shape)))))

(defn create-csr [manager data indptr indices shape & device]
  (let [data (if (sequential? data)
               (condp clojure.core/= (type (first data))
                 java.lang.Byte (ByteBuffer/wrap (byte-array data))
                 java.lang.Integer (IntBuffer/wrap (int-array data))
                 java.lang.Long (LongBuffer/wrap (long-array data))
                 java.lang.Float (FloatBuffer/wrap (float-array data))
                 java.lang.Double (DoubleBuffer/wrap (double-array data)))
               data)
        indptr (if (sequential? indptr) (long-array indptr) indptr)
        indices (if (sequential? indices) (long-array indices) indices)
        shape (if (sequential? shape) (new-shape shape) shape)]
    (if (nil? device)
      (.createCSR manager data indptr indices shape)
      (.createCSR manager data indptr indices shape (first device)))))

(defn is-sparse [ndarray]
  (.isSparse ndarray))

(def sparse? is-sparse)

(defn ndlist [arrays]
  (new NDList arrays))

(def new-ndlist ndlist)

(defn stack [col]
  (NDArrays/stack (NDList. (into-array col))))

(defn- get-datatype [data-type]
  (condp clojure.core/= (type data-type)
    java.lang.String (DataType/valueOf (.toUpperCase data-type))
    clojure.lang.Keyword (DataType/valueOf (.toUpperCase (name data-type)))
    DataType data-type))

(defn random-normal
  ([manager loc scale shape-col data-type]
   (let [data-type (get-datatype data-type)]
     (cond
       (instance? java.util.Collection shape-col)
       (.randomNormal manager loc scale (shape shape-col) data-type)
       (instance? ai.djl.ndarray.types.Shape shape-col)
       (.randomNormal manager loc scale shape-col data-type))))
  ([manager loc scale shape-col data-type device]
   (let [data-type (get-datatype data-type)]
     (cond
       (instance? java.util.Collection shape-col)
       (.randomNormal manager loc scale (shape shape-col) data-type device)
       (instance? ai.djl.ndarray.types.Shape shape-col)
       (.randomNormal manager loc scale shape-col data-type device))))
  ([manager shape-col]
   (cond
     (instance? java.util.Collection shape-col)
     (.randomNormal manager (shape shape-col))
     (instance? ai.djl.ndarray.types.Shape shape-col)
     (.randomNormal manager shape-col))))

(defmulti random-uniform
  (fn [manager low high shape & [data-type device]]
    (sequential? shape)))

(defmethod random-uniform true
  ([manager low high shape]
   (.randomUniform manager low high (new-shape shape)))
  ([manager low high shape data-type]
   (.randomUniform manager low high (new-shape shape) data-type))
  ([manager low high shape data-type device]
   (.randomUniform manager low high (new-shape shape) data-type device)))

(defmethod random-uniform :default
  ([manager low high shape]
   (.randomUniform manager low high shape))
  ([manager low high shape data-type]
   (.randomUniform manager low high shape data-type))
  ([manager low high shape data-type device]
   (.randomUniform manager low high shape data-type device)))


(defn + [array0 array1]
  (.add array0 array1))

(defn +! [array0 array1]
  (.addi array0 array1))

(defn - [array0 array1]
  (.sub array0 array1))

(defn -!
  "substract element wise in place"
  [array0 array1]
  (.subi array0 array1))

(defn * [array0 array1]
  (.mul array0 array1))

(defn *! [array0 array1]
  (.muli array0 array1))

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

(defn sum
  ([array]
   (.sum array))
  ([array axes]
   (.sum array (int-array axes)))
  ([array axes keep-dims]
   (.sum array (int-array axes) keep-dims)))

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

(defn to-vec [array]
  (vec (to-array array)))

(defn to-type
  "convert ndarray to data-type, available options are:
  \"int8\" \"uint8\" \"int32\" \"int64\"
  \"float16\" \"float32\" \"float64\"
  \"boolean\" \"string\" \"unknown\""
  [ndarray data-type copy]
  (condp clojure.core/= (type data-type)
    java.lang.String (.toType ndarray (DataType/valueOf (.toUpperCase data-type)) copy)
    clojure.lang.Keyword (.toType ndarray (DataType/valueOf (.toUpperCase (name data-type))) copy)
    DataType (.toType ndarray data-type copy)))

(defn set
  ([array index value]
   (cond
     (sequential? index)
     (.set array (NDIndex. (long-array index)) value)
     (string? index)
     (.set array (NDIndex. index (object-array [])) value)
     :else
     (.set array index value))))

(defn get
  ([array]
   (.get array (NDIndex. (long-array []))))
  ([array indices & more]
   (cond
     (instance? java.lang.Long indices)
     (.get array (long-array [indices]))
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

(defn get-gradient
  "Returns the gradient NDArray attached to this NDArray."
  [ndarray]
  (.getGradient ndarray))

(defn pp [array]
  (println (str array)))

(defn ndindex []
  (NDIndex.))

(def new-ndindex ndindex)

(defn log-softmax [ndarray axis]
  (.logSoftmax ndarray axis))
