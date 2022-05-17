(ns build
  (:refer-clojure :exclude [test])
  (:require ;;[clojure.tools.build.api :as b] ; for b/git-count-revs
            [org.corfield.build :as bb]))

(def lib 'clj-djl/clj-djl)
;; alternatively, use MAJOR.MINOR.COMMITS:
;; (def version (format "0.1.%s" (b/git-count-revs nil)))
(def version "0.1.9")

(defn test "Run the tests." [opts]
  (bb/run-tests opts))

(defn ci "Run the CI pipeline of tests (and build the JAR)." [opts]
  (-> opts
      (assoc :lib lib :version version :aliases [:test-runner])
      (bb/run-tests)
      (bb/clean)
      (bb/jar)))

(defn ci-no-test "Run the CI pipeline of tests (and build the JAR)." [opts]
  (-> opts
      (assoc :lib lib :version version)
      (bb/clean)
      (bb/jar)))

(def jar ci-no-test)

(defn install "Install the JAR locally." [opts]
  (-> opts
      (assoc :lib lib :version version)
      (bb/install)))

(defn deploy "Deploy the JAR to Clojars." [opts]
  (-> opts
      (assoc :lib lib :version version)
      (bb/deploy)))
