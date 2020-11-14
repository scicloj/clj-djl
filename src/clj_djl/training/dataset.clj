(ns clj-djl.training.dataset
  (:import [ai.djl.training.dataset Dataset
            ArrayDataset ArrayDataset$Builder]
           [ai.djl.ndarray NDArray]
           [ai.djl.training.dataset Dataset$Usage BatchSampler SequenceSampler RandomSampler
            Batch]))

(defn set-sampling
  ([builder sampler]
   (.setSampling builder sampler)
   builder)
  ([builder batch-size random]
   (.setSampling builder batch-size random)
   builder)
  ([builder batch-size random drop-last]
   (.setSampling builder batch-size random)
   builder))

(defn build [builder]
  (.build builder))

(defn prepare
  ([ds]
   (.prepare ds)
   ds)
  ([ds progress]
   (.prepare ds progress)
   ds))

(defn array-dataset-builder []
  (ArrayDataset$Builder.))

(defn new-array-dataset-builder []
  (ArrayDataset$Builder.))

(defn set-data [builder & data]
  (.setData builder (into-array NDArray data))
  builder)

(defn get-data [ds manager]
  (.getData ds manager))

(defn iter-seq
  ([iterable]
   (iter-seq iterable (.iterator iterable)))
  ([iterable iter]
   (lazy-seq
    (when (.hasNext iter)
      (cons (.next iter) (iter-seq iterable iter))))))

(defn get-data-iterator [ds manager]
  (iter-seq (get-data ds manager)))


(defn opt-labels [builder & labels]
  (.optLabels builder (into-array NDArray labels))
  builder)

(defn get-batch-data [batch]
  (.getData batch))

(defn get-batch-labels [batch]
  (.getLabels batch))

(defn close-batch [batch]
  (.close batch))

(defmulti opt-usage
  (fn [builder usage]
    (type usage)))

(defmethod opt-usage ai.djl.training.dataset.Dataset$Usage
  [builder usage]
  (.optUsage builder usage)
  builder)

(defmethod opt-usage clojure.lang.Keyword
  [builder usage]
  (let [usage-map {:test Dataset$Usage/TEST   :TEST Dataset$Usage/TEST
                   :train Dataset$Usage/TRAIN :TRAIN Dataset$Usage/TRAIN
                   :validation Dataset$Usage/VALIDATION :VALIDATION Dataset$Usage/VALIDATION}]
    (.optUsage builder (usage-map usage))
    builder))

(defn batch-sampler
  "Creates a new instance of BatchSampler that samples from the given SubSampler,
  and yields a mini-batch of batchsize, with optional droplast(true, false) to
  drop the remaining samples."
  ([subsampler batchsize]
   (BatchSampler. subsampler batchsize))
  ([subsampler batchsize droplast]
   (BatchSampler. subsampler batchsize droplast)))

(defn sequence-sampler
  "SequenceSampler samples the data from [0, dataset.size) sequentially."
  []
  (SequenceSampler.))

(defn random-sampler
  "Creates a new instance of RandomSampler with an optional seed"
  ([]
   (RandomSampler.))
  ([seed]
   (RandomSampler. seed)))

#_(defn array-dataset [{:keys [data sampler
                             labels data-batchfier device executor label-batchfier
                             limit pipeline target-pipeline]}]
  (cond-> (ArrayDataset$Builder.)
    (sequential? data) (.setData (into-array ai.djl.ndarray.NDArray data))
    sampler (if (instance? ai.djl.training.dataset.Sampler sampling)
              (.setSampling sampler)
              (if (nil? (sampler :drop-last))
                (.setSampling (:batch-size sampler) (:random sampler))
                (.setSampling (:batch-size sampler) (:random sampler) (:drop-last sampler))))
    (sequential? labels) (.optLabels (into-array ai.djl.ndarray.NDArray labels))
    data-batchfier (.optDataBatchifier data-batchfier)
    device (.optDevice device)
    ))

(defn to-apair
  "Convert dataset to a pair of two arrays. First item is the data, and the second
  item is the labels"
  [dataset]
  (let [pair (.toArray dataset)]
    [(.getKey pair) (.getValue pair)]))

(defn random-split
  [dataset & ratios]
  (.randomSplit dataset (int-array ratios)))

#_(defn new-batch [manager data-list label-list size data-batchifier label-batchifier progress progress-total]
  (Batch. manager data-list label-list size data-batchifier label-batchifier progress progress-total))
