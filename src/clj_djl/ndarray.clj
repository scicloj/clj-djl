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
                            get set
                            flatten
                            sort]))

(defn base-manager []
  (NDManager/newBaseManager))

(def new-base-manager base-manager)

(defn shape
  "Create a shape instance."
  ([]
   (shape []))
  ([param1 & more]
   (cond
     (instance? Shape param1) param1
     (instance? ai.djl.util.PairList param1) (Shape. param1)
     (.isArray (class param1)) (if (some? (first more))
                                 (Shape. param1 (first more))
                                 (Shape. param1))
     (int? param1) (Shape. (long-array (cons param1 more)))
     (sequential? param1) (if (some? (first more))
                            (Shape. (long-array param1) (first more))
                            (Shape. (long-array param1)))
     (instance? NDArray param1) (.getShape param1))))

(def new-shape shape)

(defn get-device [ndarray]
  (.getDevice ndarray))

(defn default-device []
  (Device/defaultDevice))

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
  ([ndarray]
   (.size ndarray))
  ([ndarray axis]
   (.size ndarray axis)))

(defn scalar? [ndarray]
  (.isScalar ndarray))

(defn datatype [datatype-]
  (condp clojure.core/= (type datatype-)
    java.lang.String (DataType/valueOf (.toUpperCase datatype-))
    clojure.lang.Keyword (DataType/valueOf (.toUpperCase (name datatype-)))
    datatype-))

(defn zeros
  ([manager shape-]
   (.zeros manager (shape shape-)))
  ([manager shape- datatype-]
   (.zeros manager (shape shape-) (datatype datatype-)))
  ([manager shape- datatype- device]
   (.zeros manager (shape shape-) (datatype datatype-) device)))

(defn zeros-like
  [ndarray]
  (.zerosLike ndarray))

(defn ones
  ([manager shape-]
   (.ones manager (shape shape-)))
  ([manager shape- datatype-]
   (.ones manager (shape shape-) (datatype datatype-)))
  ([manager shape- datatype- device]
   (.ones manager (shape shape-) (datatype datatype-) device)))

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
  ([manager start stop step datatype-]
   (.arange manager (float start) (float stop) (float step) (datatype datatype-)))
  ([manager start stop step datatype- device]
   (.arange manager (float start) (float stop) (float step) (datatype datatype-) device)))

(defmethod arange java.lang.Long
  ([manager stop]
   (.arange manager (int stop)))
  ([manager start stop]
   (.arange manager (int start) (int stop)))
  ([manager start stop step]
   (.arange manager (int start) (int stop) (int step)))
  ([manager start stop step datatype-]
   (.arange manager (int start) (int stop) (int step) (datatype datatype-)))
  ([manager start stop step datatype- device]
   (.arange manager (int start) (int stop) (int step) (datatype datatype-) device)))

(defmethod arange :default
  ([manager stop]
   (.arange manager stop))
  ([manager start stop]
   (.arange manager start stop))
  ([manager start stop step]
   (.arange manager start stop step))
  ([manager start stop step datatype-]
   (.arange manager (int start) (int stop) (int step) (datatype datatype-)))
  ([manager start stop step datatype- device]
   (.arange manager (int start) (int stop) (int step) (datatype datatype-) device)))


(defmulti create
  (fn [manager data & more]
    (cond
      (nil? data) :nil
      (.isArray (.getClass data)) :array
      (or (number? data) (boolean? data) (string? data)) :primitive
      (sequential? data) :sequential
      (instance? ai.djl.ndarray.types.Shape data) :shape
      ;;(instance? tech.ml.dataset.impl.dataset.Dataset data) :dataset
      )))

(defmethod create :nil
  ([manager data]
   (.create manager (shape)))
  ([manager data shape-]
   (.create manager (shape shape-))))

(defmethod create :array
  ([manager data]
   (.create manager data))
  ([manager data shape-]
   (.create manager data (shape shape-))))

(defmethod create :primitive
  [manager data]
  (.create manager data))

(defmethod create :sequential
  ([manager data]
   (let [shape- (shape (matrix/shape data))]
     (create manager data shape-)))
  ([manager param1 param2]
   (let [flat (clojure.core/flatten param1)
         shape- (if (sequential? param2)
                  (shape param2)
                  param2)]
     (condp clojure.core/= (type (first flat))
       java.lang.Boolean (.create manager (boolean-array flat) shape-)
       java.lang.Byte (.create manager (byte-array flat) shape-)
       java.lang.Integer (.create manager (int-array flat) shape-)
       java.lang.Short (.create manager (int-array flat) shape-)
       java.lang.Long (.create manager (long-array flat) shape-)
       java.lang.Float (.create manager (float-array flat) shape-)
       java.lang.Double (.create manager (double-array flat) shape-)))))

(defmethod create :shape
  ([manager shape-]
   (.create manager shape-))
  ([manager shape- datatype-]
     (.create manager shape- (datatype datatype-)))
  ([manager shape- datatype- device]
   (.create manager shape- (datatype datatype-) device)))

#_(defmethod create :dataset
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

(defn create-csr [manager data indptr indices shape- & device]
  (let [data (if (sequential? data)
               (condp clojure.core/= (type (first data))
                 java.lang.Byte (ByteBuffer/wrap (byte-array data))
                 java.lang.Integer (IntBuffer/wrap (int-array data))
                 java.lang.Long (LongBuffer/wrap (long-array data))
                 java.lang.Float (FloatBuffer/wrap (float-array data))
                 java.lang.Double (DoubleBuffer/wrap (double-array data))
                 (float-array data))
               data)
        indptr (if (sequential? indptr) (long-array indptr) indptr)
        indices (if (sequential? indices) (long-array indices) indices)]
    (if (nil? device)
      (.createCSR manager data indptr indices (shape shape-))
      (.createCSR manager data indptr indices (shape shape-) (first device)))))

(defn create-row-sparse [manager data datashape indices shape- & device]
  (let [data (if (sequential? data)
               (condp clojure.core/= (type (first data))
                 java.lang.Byte (ByteBuffer/wrap (byte-array data))
                 java.lang.Integer (IntBuffer/wrap (int-array data))
                 java.lang.Long (LongBuffer/wrap (long-array data))
                 java.lang.Float (FloatBuffer/wrap (float-array data))
                 java.lang.Double (DoubleBuffer/wrap (double-array data))
                 (float-array data))
               data)
        indices (if (sequential? indices) (long-array indices) indices)]
    (if (nil? device)
      (.createRowSparse manager data (shape datashape) indices (shape shape-))
      (.createRowSparse manager data (shape datashape) indices (shape shape-) (first device)))))

(defn is-sparse [ndarray]
  (.isSparse ndarray))

(def sparse? is-sparse)

(defn duplicate [ndarray]
  (.duplicate ndarray))

(def dup duplicate)

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

(defmethod stack java.util.Collection
  [ndarrays & [axis]]
  (if (nil? axis)
    (NDArrays/stack (ndlist ndarrays))
    (NDArrays/stack (ndlist ndarrays) axis)))

(defmethod stack clojure.lang.PersistentVector
  [coll & [axis]]
  (if (nil? axis)
    (NDArrays/stack (NDList. (into-array coll)))
    (NDArrays/stack (NDList. (into-array coll)) axis)))

(defn- get-datatype [ndarray]
  (.getDataType ndarray))

(defn random-normal
  ([manager shape-]
   (.randomNormal manager (shape shape-)))
  ([manager loc scale shape-]
   (random-normal manager loc scale (shape shape-) :float32))
  ([manager loc scale shape- datatype-]
   (.randomNormal manager loc scale (shape shape-) (datatype datatype-)))
  ([manager loc scale shape- datatype- device]
   (.randomNormal manager loc scale (shape shape-) (datatype datatype-) device)))

(defn random-uniform
  ([manager low high shape-]
   (.randomUniform manager low high (shape shape-)))
  ([manager low high shape- datatype-]
   (.randomUniform manager low high (shape shape-) (datatype datatype-)))
  ([manager low high shape- datatype- device]
   (.randomUniform manager low high (shape shape-) (datatype datatype-) device)))

(defn random-multinomial
  "Draw samples from a multinomial distribution. "
  ([ndmanager n ndarray]
   (.randomMultinomial ndmanager n ndarray))
  ([ndmanager n ndarray shape-]
   (.randomMultinomial ndmanager n ndarray (shape shape-))))

(defn + [array0 array1]
  (.add array0 array1))

(defn +! [array0 array1]
  (.addi array0 array1))

(defn -
  ([array0]
   (.neg array0))
  ([array0 array1]
   (.sub array0 array1)))

(defn -!
  "substract element wise in place"
  ([array0]
   (.negi array0))
  ([array0 array1]
   (.subi array0 array1)))

(defn * [array0 array1]
  (.mul array0 array1))

(defn mul [array0 array1]
  (.mul array0 array1))

(defn *! [array0 array1]
  (.muli array0 array1))

(defn muli [array0 array1]
  (.muli array0 array1))

(defn / [array0 array1]
  (.div array0 array1))

(defn div [array0 array1]
  (.div array0 array1))

(defn ** [array0 array1]
  (.pow array0 array1))

(defn pow [array0 array1]
  (.pow array0 array1))

(defn dot [array0 array1]
  (.dot array0 array1))

(defn = [array0 array1]
  (.eq array0 array1))

(defn < [array0 param1]
  (.lt array0 param1))

(defn > [array0 param1]
  (.gt array0 param1))

(defn <= [array0 param1]
  (.lte array0 param1))

(defn >= [array0 param1]
  (.gte array0 param1))

(defn argmax
  ([^NDArray ndarray]
   (.argMax ndarray))
  ([^NDArray ndarray axis]
   (.argMax ndarray axis)))

(defn argmin
  ([^NDArray ndarray]
   (.argMin ndarray))
  ([^NDArray ndarray axis]
   (.argMin ndarray axis)))

(defn argsort
  ([^NDArray ndarray]
   (.argSort ndarray))
  ([^NDArray ndarray axis]
   (.argSort ndarray axis))
  ([^NDArray ndarray axis ascending]
   (.argSort ndarray axis ascending)))

(defn sort
  ([^NDArray ndarray]
   (.sort ndarray))
  ([^NDArray ndarray axis]
   (.sort ndarray axis)))

(defn all-close
  ([array0 array1]
   (.allClose array0 array1))
  ([array0 array1 rtol atol equal-nan]
   (.allClose array0 array1 rtol atol equal-nan)))

(defn exp [array0]
  (.exp array0))

(defn sum
  ([array]
   (.sum array))
  ([array axes]
   (let [local-axes (cond
                      (number? axes) (int-array [axes])
                      (sequential? axes) (int-array axes)
                      :else axes)]
        (.sum array local-axes)))
  ([array axes keep-dims]
   (let [local-axes (cond
                      (number? axes) (int-array [axes])
                      (sequential? axes) (int-array axes)
                      :else axes)]
     (.sum array local-axes keep-dims))))

(defn cumsum
  ([array]
   (.cumSum array))
  ([array axis]
   (.cumSum array axis)))

(defmulti concat (fn [param1 & [param2]]
                  (type param1)))

(defmethod concat ai.djl.ndarray.NDArray
  [ndarray1 ndarray2 & [axis]]
  (if (nil? axis)
    (.concat ndarray1 ndarray2)
    (.concat ndarray1 ndarray2 axis)))

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
  "get the wrap value of NDArray"
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

(defn to-array [ndarray-or-shape]
  (cond
    (instance? NDArray ndarray-or-shape) (.toArray ndarray-or-shape)
    (instance? Shape ndarray-or-shape) (.getShape ndarray-or-shape)))

(defn to-vec [ndarray-or-shape]
  (vec (to-array ndarray-or-shape)))

(defn to-type
  "convert ndarray to data-type, available options are:
  \"int8\" \"uint8\" \"int32\" \"int64\"
  \"float16\" \"float32\" \"float64\"
  \"boolean\" \"string\" \"unknown\""
  [ndarray datatype- copy]
  (.toType ndarray (datatype datatype-) copy))

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
     (cond
       (clojure.core/fn? val-or-fun)
       (let [value (val-or-fun (get array local-index))]
         (.set array local-index value))
       (sequential? val-or-fun)
       (.set array local-index
             (.toType (create (.getManager array) val-or-fun) (.getDataType array) false))
       :else
       (.set array local-index val-or-fun))
     array)))


(defn singleton-or-throw [ndlist]
  (.singletonOrThrow ndlist))

(defn head [ndlist]
  (.head ndlist))

(defn set-requires-gradient
  "Attaches a gradient NDArray to this NDArray and marks it so
  GradientCollector.backward(NDArray) can compute the gradient with respect to it."
  [ndarray requires-grad]
  (.setRequiresGradient ndarray requires-grad))

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

(defn log10 [ndarray]
  (.log10 ndarray))

(defn log [ndarray]
  (.log ndarray))

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

(defn norm
  ([ndarray]
   (.sqrt (.dot ndarray ndarray))))

(defn abs
  "Returns the absolute value of this NDArray element-wise"
  [ndarray]
  (.abs ndarray))

(defn full
  ([manager shape- data]
   (let [data (if (float? data) (float data) data)]
     (.full manager (shape shape-) data)))
  ([manager shape- data datatype-]
   (let [data (if (float? data) (float data) data)]
     (.full manager (shape shape-) data (datatype datatype-))))
  ([manager shape- data datatype- device]
   (let [data (if (float? data) (float data) data)]
     (.full manager (shape shape-) data (datatype datatype-) device))))

(defn eye
  ([manager rows]
   (.eye manager rows))
  ([manager rows k]
   (.eye manager rows k))
  ([manager rows cols k]
   (.eye manager rows cols k))
  ([manager rows cols k datatype-]
   (.eye manager rows cols k (datatype datatype-)))
  ([manager rows cols k datatype- device]
   (.eye manager rows cols k (datatype datatype-) device)))

(defn float-or-int
  [n]
  (cond
    (float? n) (float n)
    (int? n) (int n)
    :else n))

(defn linspace
  ([manager start stop n]
   (.linspace manager start stop n))
  ([manager start stop n endpoint]
   (.linspace manager start stop n endpoint))
  ([manager start stop n endpoint device]
   (.linspace manager start stop n endpoint device)))

(defn acos
  "Returns the inverse trigonometric cosine of this NDArray element-wise."
  [ndarray]
  (.acos ndarray))

(defn sqrt
  [ndarray]
  (.sqrt ndarray))
