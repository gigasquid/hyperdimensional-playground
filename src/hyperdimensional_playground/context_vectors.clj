(ns hyperdimensional-playground.context-vectors
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.linear :as ml]
            [clojure.java.io :as io]
            [clojure.string :as string])
  (:import (java.util Properties)
           (edu.stanford.nlp.pipeline StanfordCoreNLP Annotation)
           (edu.stanford.nlp.ling CoreAnnotations$SentencesAnnotation CoreAnnotations$TokensAnnotation
                                  CoreAnnotations$TextAnnotation CoreAnnotations$PartOfSpeechAnnotation
                                  CoreAnnotations$NamedEntityTagAnnotation)))

(defn rand-hv []
  (let [hv (m/new-sparse-array [sz])
        n (* 0.1 sz)]
    (dotimes [i n]
      (m/mset! hv (rand-int sz) 1))
    hv))

(defn cosine-sim [v1 v2]
  (let [norm1 (ml/norm v1)
        norm2 (ml/norm v2)]
   (when (and (pos? norm1) ( pos? norm2))
     (/ (m/dot v1 v2)
        (* (ml/norm v1) (ml/norm v2))))))

(defn mean-add [& hvs]
  (m/emap #(Math/round %)
   (m/div (apply m/add hvs) (count hvs))))

(defn xor-mul [v1 v2]
  (->> (m/add v1 v2)
      (m/emap #(mod % 2))))

(defn hamming-dist [v1 v2]
  (m/esum (xor-mul v1 v2)))

(defn inverse [hv]
  (m/emap #(- 1 %) hv))



;;; get all the nouns in our corpus
(def props (Properties.))
(.setProperty props "annotators", "tokenize, ssplit, pos")

(def pipeline (StanfordCoreNLP. props))

(defn ->text-data [tokens sent-num]
  (mapv (fn [t] {:sent-num sent-num
                :token (.get t CoreAnnotations$TextAnnotation)
                :pos  (.get t CoreAnnotations$PartOfSpeechAnnotation)}) tokens))

(defn process-text [text]
  (let [annotation (Annotation. text)
        _ (.annotate pipeline annotation)
        sentences (.get annotation CoreAnnotations$SentencesAnnotation)
        sentence-tokens (mapv (fn [s] (.get s CoreAnnotations$TokensAnnotation)) sentences)
        text-data (flatten (map-indexed (fn [i t] (->text-data t i)) sentence-tokens))]
    text-data))

(defn filter-nouns [doc]
  (let [nouns (filter #(contains? (hash-set "NN" "NNS" "NNP" "NNPS") (:pos %))
                      (process-text doc))]
    (remove #{"and" "the" "is" "a" "i" "he" "she" "it"}
            (set (map #(clojure.string/lower-case (:token %)) nouns)))))

(defn gather-nouns-from-book [book-str]
  (let [book-text (slurp book-str)
        docs (string/split book-text #"\s|\.|\,|\;|\!|\?")]
    (doall (reduce (fn [nouns doc]
               (clojure.set/union nouns (filter-nouns doc)))
             #{}
             docs))))
(comment
  (def grimm-nouns (gather-nouns-from-book "resources/grimm_fairy_tales.txt"))
  (def anderson-nouns (gather-nouns-from-book "resources/anderson_fairy_tales.txt"))
  (def english-fairy-tale-nouns (gather-nouns-from-book "resources/english_fairy_tales.txt"))
  (def every-child-nouns (gather-nouns-from-book "resources/fairy_tales_every_child_should_know.txt"))
  (def firelight-nouns (gather-nouns-from-book "resources/firelight_fairy_book.txt"))
  (def favorite-nouns (gather-nouns-from-book "resources/favorite_fairy_tales.txt"))
  (def wonder-wings-nouns (gather-nouns-from-book "resources/wonder_wings.txt"))
  (def old-fashioned-nouns (gather-nouns-from-book "resources/old_fashioned.txt"))
  (def fairy-godmother-nouns (gather-nouns-from-book "resources/fairy_godmothers.txt"))
  (def golden-spears-nouns (gather-nouns-from-book "resources/golden_spears.txt"))
  (def red-cap-nouns (gather-nouns-from-book "resources/red_cap_tales.txt"))


  (def fairy-tales-nouns (clojure.set/union grimm-nouns anderson-nouns english-fairy-tale-nouns
                                            every-child-nouns firelight-nouns favorite-nouns
                                            wonder-wings-nouns old-fashioned-nouns fairy-godmother-nouns
                                            golden-spears-nouns red-cap-nouns))

  (take 100 grimm-nouns)
  (count grimm-nouns)
  (count fairy-tales-nouns) ;=> 136606
  (spit "resources/fairy-tales-nouns.edn" fairy-tales-nouns)
  (spit "resources/fairy-tales-nouns-grimm.edn" grimm-nouns)
  (take 100 fairy-tales-nouns)

 (def x (clojure.edn/read-string (slurp "resources/fairy-tales-nouns.edn")))

)
(def grimm-nouns (clojure.edn/read-string (slurp "resources/fairy-tales-nouns-grimm.edn")))
(def fairy-tales-nouns (set grimm-nouns))


;;;; construct a matrix of 10322 x 10,000 and use random columns of 20
;;;; to put frequenies

(def book-list ["resources/grimm_fairy_tales.txt"
                "resources/anderson_fairy_tales.txt"
                "resources/english_fairy_tales.txt"
                "resources/fairy_tales_every_child_should_know.txt"
                "resources/firelight_fairy_book.txt"
                "resources/favorite_fairy_tales.txt"
                "resources/wonder_wings.txt"
                "resources/old_fashioned.txt"
                "resources/fairy_godmothers.txt"
                "resources/golden_spears.txt"
                "resources/red_cap_tales.txt"])


(m/set-current-implementation :vectorz)
(def sz 10000)
(def noun-idx (zipmap fairy-tales-nouns (range)))
(def freq-matrix (m/new-sparse-array [(count fairy-tales-nouns) sz]))


(defn update-doc! [doc]
  (let [known-nouns (clojure.set/intersection fairy-tales-nouns (set doc))]
    ; use positive indexing
    (doall (repeatedly 10 #(doseq [noun known-nouns]
                       (m/mset! freq-matrix (get noun-idx noun) (rand-int sz) 1))))))

(defn process-book [book-str]
  (let [book-text (slurp book-str)
        docs (partition 25 (map string/lower-case
                                (string/split book-text #"\s|\.|\,|\;|\!|\?")))
        doc-count (count docs)]
    (println "processing:" book-str "(with" doc-count "docs)")
    (doall (map-indexed (fn [idx doc]
                          (when (zero? (mod idx 1000)) (println "doc:" idx))
                          (update-doc! doc))
                        docs))
    (println "DONE with " book-str)))


(defn get-word-vec [word]
  (let [i (get noun-idx word)]
    (assert (not (nil? i)) (str word " not found"))
    (m/slice freq-matrix i)))

(defn compare-word-vecs [word1 word2]
  (let [i1 (get noun-idx word1)
        i2 (get noun-idx word2)]
    (assert (not (nil? i1)) (str word1 " not found"))
    (assert (not (nil? i2)) (str word2 " not found"))
    (when (not= word1 word2)
      {:word1 word1
       :word2 word2
       :cosine (cosine-sim (m/slice freq-matrix i1)
                           (m/slice freq-matrix i2))})))

(defn sim-report []
  (for [word1 (take 100 fairy-tales-nouns)
        word2 (take 100 fairy-tales-nouns)]
    (when (not= word1 word2)
      {:word1 word1
       :word2 word2
       :cosine (cosine-sim (m/slice freq-matrix (get noun-idx word1))
                           (m/slice freq-matrix (get noun-idx word2)))})))


(defn sim-for-word [word threshold]
  (let [results  (mapv #(compare-word-vecs word %)
                       fairy-tales-nouns)]
    (->> results
        (filter #(and (:cosine %) (< threshold (:cosine %))))
        (sort-by :cosine)
        (reverse))))

(defn sim-for-vec [wv threshold]
  (let [results  (mapv #(hash-map :word %
                                 :cosine (cosine-sim wv (get-word-vec %)))
                       fairy-tales-nouns)]
    (->> results
        (filter #(and (:cosine %) (< threshold (:cosine %))))
        (sort-by :cosine)
        (reverse))))


(defn unlike-for-word [word threshold]
  (let [results  (mapv #(compare-word-vecs word %)
                       fairy-tales-nouns)]
    (->> results
        (filter #(and (:cosine %) (> threshold (:cosine %))))
        (sort-by :cosine))))

(defn wv-subtract [word1 word2]
  (mean-add (get-word-vec word1) (inverse (get-word-vec word2))))

(defn wv-add [word1 word2]
  (mean-add (get-word-vec word1) (get-word-vec word2)))

(comment

  (doseq [book book-list]
    (process-book book))



                                        ;(process-book "resources/grimm_fairy_tales.txt")

  (def x (remove nil? (sim-report)))
  (second x)
  (take 100 x)
  (sort-by :cosine (filter #( :cosine % ) x))

  (def results (sort-by :cosine (filter #(and (:cosine %) (< 0.4 (:cosine %))) x)))


  (take 100 (reverse results))

  (sim-for-word "fairy" 0.4)
  (sim-for-word "prince" 0.5)
  (sim-for-word "king" 0.01)
  (sim-for-word "queen" 0.4)
  (sim-for-word "witch" 0.1)
  (sim-for-word "heart" 0.1)
  (sim-for-word "apples" 0.1)
  (sim-for-word "guard" 0.1)
  (sim-for-word "eyes" 0.2)
  (sim-for-word "monster" 0.4)
  (sim-for-word "time" 0.6)
  (unlike-for-word "king" 0.04) ;=> "tusks"
  (sim-for-word "tusks" 0.01)
  (sim-for-word "goat" 0.1)
  (sim-for-vec (get-word-vec "fairy") 0.5)

  (sort-by :cosine[(compare-word-vecs "king" "queen")
                   (compare-word-vecs "king" "prince")
                   (compare-word-vecs "king" "princess")
                   (compare-word-vecs "king" "guard")
                   (compare-word-vecs "king" "goat")])
  ;; ({:word1 "king", :word2 "goat", :cosine 0.1509151478896664}
  ;;  {:word1 "king", :word2 "guard", :cosine 0.16098893367403827}
  ;;  {:word1 "king", :word2 "queen", :cosine 0.49470535530616655}
  ;;  {:word1 "king", :word2 "prince", :cosine 0.5832521795716931}
  ;;  {:word1 "king", :word2 "princess", :cosine 0.5836922474743367})



  (cosine-sim (get-word-vec "boy") (get-word-vec "king")) ;=> 0.42996397142253145
  (cosine-sim (get-word-vec "boy") (get-word-vec "goat")) ;=> 0.10263384987428588
  (cosine-sim
   (mean-add (get-word-vec "boy") (get-word-vec "gold"))
   (get-word-vec "king")) ;=> 0.5876251031366048
  (sim-for-vec (mean-add (get-word-vec "boy") (get-word-vec "gold")) 0.5)


  (cosine-sim (get-word-vec "girl") (get-word-vec "queen")) ;=> 0.3356337018948302
  (cosine-sim (get-word-vec "girl") (get-word-vec "goat")) ;=> 0.10061278928161434
  (cosine-sim
   (mean-add (get-word-vec "girl") (get-word-vec "gold"))
   (get-word-vec "queen")) ;=> 0.437410495036485
  (cosine-sim
   (mean-add (get-word-vec "girl") (get-word-vec "prince"))
   (get-word-vec "queen")) ;=> 0.49567588785560884

  (cosine-sim (get-word-vec "queen")
              (mean-add (get-word-vec "woman") (wv-subtract "king" "man"))) ;=>0.5659832204544486

  (cosine-sim (mean-add (get-word-vec "giant") (get-word-vec "boy")) (get-word-vec "jack")) ;=>0.4491473187787431
  (cosine-sim (get-word-vec "boy") (get-word-vec "jack")) ;=> 0.33102858702785953

  (cosine-sim (wv-add "frog" "princess") (get-word-vec "prince")) ;=> 0.5231641991974249
  (cosine-sim (wv-add "frog" "witch") (get-word-vec "prince")) ;=> 0.28231826406597205
  (cosine-sim (wv-add "frog" "magic") (get-word-vec "prince")) ;=> 0.24898297475282194
  (cosine-sim (get-word-vec "frog") (get-word-vec "prince")) ;=> 0.1507782796891884
  (cosine-sim (get-word-vec "father")
              (mean-add (get-word-vec "man") (wv-subtract "mother" "woman"))) ;=>0.5959841177719538
  (cosine-sim (get-word-vec "father") (get-word-vec "woman"));=>
                                        ;0.39586439279718383

  (take 400 noun-idx)

  (compare-word-vecs "prince" "princess")
  (compare-word-vecs "apple" "tree")
  (compare-word-vecs "apple" "round")

  (compare-word-vecs "girl" "boy") ;=> {:word1 "girl", :word2 "boy", :cosine 0.5127662367009652}
  (compare-word-vecs "dragon" "fire")
  (compare-word-vecs "sea" "boat")
  (compare-word-vecs "sea" "bird")
  (compare-word-vecs "sea" "shoe")
  (compare-word-vecs "witch" "magic")


  ;; hansel is the brother of gretel
  ;; B*H + B*G
  (def hansel-brother-of-gretel
    (mean-add
     (xor-mul (get-word-vec "brother") (get-word-vec "hansel"))
     (xor-mul (get-word-vec "brother") (get-word-vec "gretel"))))
  (def jack-brother-of-hansel
    (mean-add
     (xor-mul (get-word-vec "brother") (get-word-vec "jack"))
     (xor-mul (get-word-vec "brother") (get-word-vec "hansel"))))

  (def facts (mean-add hansel-brother-of-gretel
                       jack-brother-of-hansel))

  ;; is hansel the brother of gretel?

  (cosine-sim
   (get-word-vec "hansel")
   (xor-mul (mean-add (get-word-vec "brother") (get-word-vec "gretel"))
            facts)) ;=> 0.542518819404095

  ;; is john the brother of gretel?
  (cosine-sim
   (get-word-vec "john")
   (xor-mul (mean-add (get-word-vec "brother") (get-word-vec "gretel"))
            facts)) ;=> 0.13036770808976186

  ;; is cinderella the brother of gretel ?
  (cosine-sim
   (get-word-vec "cinderella")
   (xor-mul (mean-add (get-word-vec "brother") (get-word-vec "gretel"))
            facts)) ;=> 0.12327219767046675

  ;; is jack the brother of gretel ?
  (cosine-sim
   (get-word-vec "jack")
   (xor-mul (mean-add (get-word-vec "brother") (get-word-vec "gretel"))
            facts));=> 0.3354538693328559


  ;; let's add some more facts

  ;; gretel is the sister of hansel
                                        ; S*G + S*H
  (def gretel-sister-of-hansel
    (mean-add
     (xor-mul (get-word-vec "sister") (get-word-vec "gretel"))
     (xor-mul (get-word-vec "sister") (get-word-vec "hansel"))))


  ;; gretel is the sister of jack
                                        ; S*G + S*H
  (def gretel-sister-of-jack
    (mean-add
     (xor-mul (get-word-vec "sister") (get-word-vec "gretel"))
     (xor-mul (get-word-vec "sister") (get-word-vec "jack"))))

;;; a sibling is a brother and sister

  (def siblings (rand-hv))
  (def siblings-brother-sister
    (mean-add (xor-mul siblings (get-word-vec "brother")) (xor-mul siblings (get-word-vec "sister"))))


  (def facts (mean-add hansel-brother-of-gretel
                       jack-brother-of-hansel
                       gretel-sister-of-jack
                       gretel-sister-of-hansel
                       siblings-brother-sister))

  ;; are hansel and gretel siblings?
  (cosine-sim
   (mean-add (get-word-vec "hansel") (get-word-vec "gretel"))
   (xor-mul siblings facts)) ;=> 0.6003227988919575

  ;; are john and cinderella siblings?
  (cosine-sim
   (mean-add (get-word-vec "john") (get-word-vec "cinderella"))
   (xor-mul siblings facts)) ;=> 0.26025663087008655

  ;; are jack and gretel siblings?
  (cosine-sim
   (mean-add (get-word-vec "jack") (get-word-vec "gretel"))
   (xor-mul siblings facts)) ;=> 0.5462252603770413

  ;; are jack and hansel siblings?
  (cosine-sim
   (mean-add (get-word-vec "jack") (get-word-vec "hansel"))
   (xor-mul siblings facts)) ;=> 0.5431168503661485

  ;; are jack and cinderella siblings?
  (cosine-sim
   (mean-add (get-word-vec "hansel") (get-word-vec "cinderella"))
   (xor-mul siblings facts)) ;=>0.44419839153931734
  )
