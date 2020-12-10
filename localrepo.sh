#!/bin/sh
export version=0.1.3
lein uberjar
lein pom
lein localrepo install target/clj-djl-$version.jar clj-djl $version
mv pom.xml ~/.m2/repository/clj-djl/clj-djl/$version/clj-djl-$version.pom
