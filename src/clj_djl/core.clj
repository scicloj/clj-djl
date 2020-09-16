(ns clj-djl.core
  (:require [clj-djl.engine :as engine]))

(defn -main
  "List available engines."
  [& args]
  (println (str (engine/get-all-engines))))
