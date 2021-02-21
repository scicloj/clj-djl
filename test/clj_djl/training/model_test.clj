(ns clj-djl.training.model-test
  (:require [clojure.test :refer [deftest is]]
            [clj-djl.nn :as nn]
            [clj-djl.model :as m]
            [clj-djl.training.initializer :as initializer]))

(deftest model-test
  (let [block (-> (nn/sequential-block)
                  (nn/add (nn/cov2d-block {:kernel-shape [1 1] :filters 10}))
                  (nn/add (nn/batchnorm-block))
                  (nn/set-initializer (initializer/new-xavier)))]
    (with-open [savemodel (m/new-model {:name "save-model"})
                loadmodel (m/new-model {:name "load-model"})]
      (nn/initialize block (m/get-ndmanager savemodel) :float32 [1 3 32 32])
      (let [saved-parameters (nn/get-parameters block)]
        (m/set-block savemodel block)
        (m/save savemodel "build/tmp/test/models" "save-and-load")
        (nn/clear block)
        (m/set-block loadmodel block)
        (m/load loadmodel "build/tmp/test/models" "save-and-load")
        (let [load-parameters (nn/get-parameters block)]
          (is (= (.size load-parameters) (.size saved-parameters)))
          (is (reduce #(and %1 %2)
                      (map #(= (.getArray (.getValue %1)) (.getArray (.getValue %2)))
                           saved-parameters load-parameters))))))))
