(ns clj-djl.ndarray
  (:require [clojure.core.matrix :as matrix]
            [tablecloth.api :as tablecloth])
  (:import [ai.djl.ndarray NDManager NDArray NDList NDArrays]
           [ai.djl.ndarray.index NDIndex]
           [ai.djl.ndarray.types Shape DataType]
           [ai.djl Device]
           [java.nio ByteBuffer IntBuffer LongBuffer FloatBuffer DoubleBuffer])
  (:refer-clojure :exclude [+ - / *
                            = <= < >= >
                            identity to-array
                            min max concat
                            get set
                            flatten]))

(defn new-base-manager []
  (NDManager/newBaseManager))

(def base-manager new-base-manager)

(def manager (NDManager/newBaseManager))

(defn get-device [ndarray]
  (.getDevice ndarray))

(defn default-device []
  (Device/defaultDevice))

(defn shape
  ([]
   (shape []))
  ([param1 & more]
   (cond
     (instance? ai.djl.util.PairList param1) (Shape. param1)
     (.isArray (class param1)) (if (some? (first more))
                                 (Shape. param1 (first more))
                                 (Shape. param1))
     (int? param1) (Shape. (long-array (cons param1 more)))
     (sequential? param1) (if (some? (first more))
                            (Shape. (long-array param1) (first more))
                            (Shape. (long-array param1))))))

(def new-shape shape)

(defn get-shape [ndarray]
  (.getShape ndarray))

(defn reshape
  ([ndarray]
   (.reshape ndarray))
  ([ndarray param1 & more]
   (cond
     (instance? ai.djl.ndarray.types.Shape param1) (.reshape ndarray param1)
     (int? param1) (.reshape ndarray (long-array (cons param1 more)))
     (sequential? param1) (.reshape ndarray (long-array param1)))))

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
(defn- convert-datatype [data-type]
  (condp clojure.core/= (type data-type)
    java.lang.String (DataType/valueOf (.toUpperCase data-type))
    clojure.lang.Keyword (DataType/valueOf (.toUpperCase (name data-type)))
    data-type))

(defn zeros
  ([manager shape]
   (let [local-shape (if (sequential? shape) (new-shape shape) shape)]
     (.zeros manager local-shape)))
  ([manager shape data-type]
   (let [local-data-type (if (keyword? data-type) (datatype-map data-type) data-type)
         local-shape (if (sequential? shape) (new-shape shape) shape)]
     (.zeros manager local-shape local-data-type)))
  ([manager shape data-type device]
   (let [local-data-type (if (keyword? data-type) (datatype-map data-type) data-type)
         local-shape (if (sequential? shape) (new-shape shape) shape)]
     (.zeros manager local-shape local-data-type)) device))

(defn zeros-like
  [ndarray]
  (.zerosLike  ndarray))

(defn ones
  ([manager shape]
   (let [local-shape (if (sequential? shape) (new-shape shape) shape)]
     (.ones manager local-shape)))
  ([manager shape data-type]
   (let [local-data-type (if (keyword? data-type) (datatype-map data-type) data-type)
         local-shape (if (sequential? shape) (new-shape shape) shape)]
     (.ones manager local-shape local-data-type)))
  ([manager shape data-type device]
   (let [local-data-type (if (keyword? data-type) (datatype-map data-type) data-type)
         local-shape (if (sequential? shape) (new-shape shape) shape)]
     (.ones manager local-shape local-data-type)) device))

(defn ones-like
  [ndarray]
  (.onesLike ndarray))

(defmulti arange (fn [manager start-or-stop & more]
                   (type start-or-stop)))

(defmethod arange java.lang.Double
  ([manager stop]
   (.arange manager (float stop)))
  ([manager start stop]
   (.arange manager (float start) (float stop)))
  ([manager start stop step]
   (.arange manager (float start) (float stop) (float step)))
  ([manager start stop step data-type]
   (condp clojure.core/= (type data-type)
     java.lang.String (.arange manager (float start) (float stop) (float step)
                               (DataType/valueOf (.toUpperCase data-type)))
     clojure.lang.Keyword (.arange manager (float start) (float stop) (float step)
                                   (DataType/valueOf (.toUpperCase (name data-type))))
     DataType (.arange manager (float start) (float stop) (float step) data-type)))
  ([manager start stop step data-type device]
   (condp clojure.core/= (type data-type)
     java.lang.String (.arange manager (float start) (float stop) (float step)
                               (DataType/valueOf (.toUpperCase data-type)))
     clojure.lang.Keyword (.arange manager (float start) (float stop) (float step)
                                   (DataType/valueOf (.toUpperCase (name data-type))) device)
     DataType (.arange manager (float start) (float stop) (float step) data-type device))))

(defmethod arange java.lang.Long
  ([manager stop]
   (.arange manager (int stop)))
  ([manager start stop]
   (.arange manager (int start) (int stop)))
  ([manager start stop step]
   (.arange manager (int start) (int stop) (int step)))
  ([manager start stop step data-type]
   (condp clojure.core/= (type data-type)
     java.lang.String (.arange manager (int start) (int stop) (int step)
                               (DataType/valueOf (.toUpperCase data-type)))
     clojure.lang.Keyword (.arange manager (int start) (int stop) (int step)
                                   (DataType/valueOf (.toUpperCase (name data-type))))
     DataType (.arange manager (int start) (int stop) (int step) data-type)))
  ([manager start stop step data-type device]
   (condp clojure.core/= (type data-type)
     java.lang.String (.arange manager (int start) (int stop) (int step)
                               (DataType/valueOf (.toUpperCase data-type)))
     clojure.lang.Keyword (.arange manager (int start) (int stop) (int step)
                                   (DataType/valueOf (.toUpperCase (name data-type))) device)
     DataType (.arange manager (int start) (int stop) (int step) data-type device))))

(defmethod arange :default
  ([manager stop]
   (.arange manager stop))
  ([manager start stop]
   (.arange manager start stop))
  ([manager start stop step]
   (.arange manager start stop step))
  ([manager start stop step data-type]
   (condp clojure.core/= (type data-type)
     java.lang.String (.arange manager (int start) (int stop) (int step)
                               (DataType/valueOf (.toUpperCase data-type)))
     clojure.lang.Keyword (.arange manager (int start) (int stop) (int step)
                                   (DataType/valueOf (.toUpperCase (name data-type))))
     DataType (.arange manager (int start) (int stop) (int step) data-type)))
  ([manager start stop step data-type device]
   (condp clojure.core/= (type data-type)
     java.lang.String (.arange manager (int start) (int stop) (int step)
                               (DataType/valueOf (.toUpperCase data-type)))
     clojure.lang.Keyword (.arange manager (int start) (int stop) (int step)
                                   (DataType/valueOf (.toUpperCase (name data-type))) device)
     DataType (.arange manager (int start) (int stop) (int step) data-type device))))


(defmulti create
  (fn [manager data & more]
    (cond
      (.isArray (.getClass data)):array
      (or (number? data) (boolean? data) (string? data)) :primitive
      (sequential? data) :sequential
      (instance? ai.djl.ndarray.types.Shape data) :shape
      (instance? tech.ml.dataset.impl.dataset.Dataset data) :dataset)))

(defmethod create :array
  ([manager data]
   (.create manager data))
  ([manager data shape]
   (let [local-shape (if (sequential? shape) (new-shape shape) shape)]
     (.create manager data local-shape))))

(defmethod create :primitive
  [manager data]
  (.create manager data))

(defmethod create :sequential
  ([manager data]
   (let [param-shape (new-shape (matrix/shape data))]
     (create manager data param-shape)))
  ([manager param1 param2]
   (let [flat (clojure.core/flatten param1)
         param-shape (if (sequential? param2)
                       (shape param2)
                       param2)]
     (condp clojure.core/= (type (first flat))
       java.lang.Boolean (.create manager (boolean-array flat) param-shape)
       java.lang.Byte (.create manager (byte-array flat) param-shape)
       java.lang.Integer (.create manager (int-array flat) param-shape)
       java.lang.Short (.create manager (int-array flat) param-shape)
       java.lang.Long (.create manager (long-array flat) param-shape)
       java.lang.Float (.create manager (float-array flat) param-shape)
       java.lang.Double (.create manager (double-array flat) param-shape)))))

(defmethod create :shape
  ([manager shape]
   (.create manager shape))
  ([manager shape data-type]
   (let [local-data-type (convert-datatype data-type)]
     (.create manager shape local-data-type)))
  ([manager shape data-type device]
   (let [local-data-type (convert-datatype data-type)]
     (.create manager shape local-data-type device))))

(defmethod create :dataset
  [manager ds]
  (let [data (if (clojure.core/= 1 (tablecloth/column-count ds))
               (clojure.core/flatten (tablecloth/rows ds))
               (map vec (tablecloth/rows ds)))]
    (create manager data)))

(defmethod create :default
  ([manager data]
   (.create manager data))
  ([manager data1 data2]
   (.create manager data1 data2))
  ([manager data1 data2 data3]
   (.create manager data1 data2 data3)))

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

(defmulti ndlist (fn [& more]
                   (type (first more))))

(defmethod ndlist nil
  []
  (NDList.))

(defmethod ndlist ai.djl.ndarray.NDArray
  [& more]
  (NDList. (into-array NDArray more)))

(defmethod ndlist java.util.Collection
  [other]
  (NDList. other))

(def new-ndlist ndlist)

(defmulti stack (fn [param1 & [param2]]
                  (type param1)))

(defmethod stack ai.djl.ndarray.NDArray
  [ndarray1 ndarray2 & [axis]]
  (if (nil? axis)
    (.stack ndarray1 ndarray2)
    (.stack ndarray1 ndarray2 axis)))

(defmethod stack ai.djl.ndarray.NDList
  [ndlist & [axis]]
  (if (nil? axis)
    (NDArrays/stack ndlist)
    (NDArrays/stack ndlist axis)))

(defmethod stack clojure.lang.PersistentVector
  [coll & [axis]]
  (if (nil? axis)
    (NDArrays/stack (NDList. (into-array coll)))
    (NDArrays/stack (NDList. (into-array coll)) axis)))

(defn- get-datatype [ndarray]
  (.getDataType ndarray))

(defn random-normal
  ([manager shape]
   (let [local-shape (if (sequential? shape) (new-shape shape) shape)]
     (.randomNormal manager local-shape)))
  ([manager loc scale shape data-type]
   (let [local-shape (if (sequential? shape) (new-shape shape) shape)
         local-data-type (if (keyword? data-type) (datatype-map data-type) data-type)]
     (.randomNormal manager loc scale local-shape local-data-type)))
  ([manager loc scale shape data-type device]
   (let [local-shape (if (sequential? shape) (new-shape shape) shape)
         local-data-type (if (keyword? data-type) (datatype-map data-type) data-type)]
     (.randomNormal manager loc scale local-shape local-data-type device))))

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

(defn cumsum
  ([array]
   (.cumSum array))
  ([array axis]
   (.cumSum array axis)))

(defmulti concat (fn [param1 & [param2]]
                  (type param1)))

(defmethod concat ai.djl.ndarray.NDArray
  [ndarray1 ndarray2 & axis]
  (if (empty? axis)
    (.concat ndarray1 ndarray2)
    (if (keyword? (first axis))
      (.concat ndarray1 ndarray2 (get axis 1))
      (.concat ndarray1 ndarray2 (first axis)))))

(defmethod concat ai.djl.ndarray.NDList
  [ndlist & [axis]]
  (if (nil? axis)
    (NDArrays/concat ndlist)
    (NDArrays/concat ndlist axis)))

(defmethod concat clojure.lang.PersistentVector
  [coll & [axis]]
  (if (nil? axis)
    (NDArrays/concat (NDList. (into-array coll)))
    (NDArrays/concat (NDList. (into-array coll)) axis)))


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

(defn to-array [ndarray]
  (.toArray ndarray))

(defn to-vec [ndarray]
  (vec (to-array ndarray)))

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

(defmulti get (fn [param & more]
                (type param)))

(defmethod get ai.djl.ndarray.NDArray
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

(defmethod get ai.djl.ndarray.NDList
  [ndlist index]
  (.get ndlist index))

(defmethod get ai.djl.ndarray.types.Shape
  [ndshape dim]
  (.get ndshape dim))


(defn set
  ([array data]
   (.set array data)
   array)
  ([array index val-or-fun]
   (let [local-index
         (cond
           (sequential? index) (NDIndex. (long-array index))
           (string? index) (NDIndex. index (object-array []))
           :else index)]
     (if (clojure.core/fn? val-or-fun)
       (let [value (val-or-fun (get array local-index))]
         (.set array local-index value))
       (.set array local-index val-or-fun))
     array)))


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

(defn ndindex
  ([]
   (NDIndex.))
  ([param1 & more]
   (cond
     (int? param1) (NDIndex. (long-array (cons param1 more)))
     (string? param1) (NDIndex. param1 (into-array java.lang.Object more)))))

(def new-ndindex ndindex)
(def index ndindex)

(defn log-softmax [ndarray axis]
  (.logSoftmax ndarray axis))

(defn split [ndarray index-section & [axis]]
  (let [local-index-section (if (sequential? index-section)
                              (long-array index-section)
                              index-section)]
    (if (nil? axis)
      (.split ndarray local-index-section)
      (.split ndarray local-index-section axis))))

(defn flatten [ndarray]
  (.flatten ndarray))

(defn expand-dims [ndarray axis]
  (.expandDims ndarray axis))

(defn squeeze [ndarray & [axis]]
  (if (nil? axis)
    (.squeeze ndarray)
    (if (number? axis)
      (.squeeze ndarray axis)
      (if (sequential? axis)
        (let [axis (int-array axis)]
          (.squeeze ndarray axis))
        (.squeeze ndarray axis)))))

(defn set-scalar [ndarray ndindex value]
  (.setScalar ndarray ndindex value))

(defn max
  [ndarray & [axes keep-dims]]
  (if-not (nil? axes)
    (let [axes (int-array axes)]
      (if-not (nil? keep-dims)
        (.max ndarray axes keep-dims)
        (.max ndarray axes)))
    (.max ndarray)))

(defn min
  [ndarray & [axes keep-dims]]
  (if-not (nil? axes)
    (let [axes (int-array axes)]
      (if-not (nil? keep-dims)
        (.min ndarray axes keep-dims)
        (.min ndarray axes)))
    (.min ndarray)))

(defn prod
  [ndarray & [axes keep-dims]]
  (if-not (nil? axes)
    (let [axes (int-array axes)]
      (if-not (nil? keep-dims)
        (.prod ndarray axes keep-dims)
        (.prod ndarray axes)))
    (.prod ndarray)))

(defn mean
  [ndarray & [axes keep-dims]]
  (if-not (nil? axes)
    (let [axes (int-array axes)]
      (if-not (nil? keep-dims)
        (.mean ndarray axes keep-dims)
        (.mean ndarray axes)))
    (.mean ndarray)))

(defn trace
  ([ndarray offset axis1 axis2]
   (.trace ndarray offset axis1 axis2))
  ([ndarray offset]
   (.trace ndarray offset))
  ([ndarray]
   (.trace ndarray)))

(defn transpose
  ([ndarray]
   (.transpose ndarray))
  ([ndarray axis & [more]]
   (if (sequential? axis)
     (.transpose ndarray (int-array axis))
     (.transpose ndarray (int-array (cons [axis] more))))))

(def t transpose)

(defn copy [ndarray]
  (create (.getManager ndarray) ndarray))
