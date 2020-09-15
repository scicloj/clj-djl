#!/bin/sh
export clj-djl-version=0.1.1
lein uberjar
lein pom
lein localrepo install target/clj-djl-$clj-djl-version.jar clj-djl $clj-djl-version
mv pom.xml ~/.m2/repository/clj-djl/clj-djl/$clj-djl-version/clj-djl-$clj-djl-version.pom
