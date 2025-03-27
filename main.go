package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
)

type DocTF struct {
	Name string
	TF   map[string]float64
}

func initTFSlice(len int) []DocTF {
	tfs := make([]DocTF, len)
	for i := 0; i < len; i++ {
		tfs[i].TF = make(map[string]float64)
	}
	return tfs
}

type Doc struct {
	Name string
	Text string
}

func initTestDocs() []Doc {
	return []Doc{
		{"doc1", "ananas banana apple"},
		{"doc2", "apple orange apple banana apple"},
		{"doc3", "grape ananas peach banana ananas"},
		{"doc4", "mango pineapple banana"},
		{"doc5", "strawberry blueberry raspberry ananas ananas"},
		{"doc6", "banana apple peach"},
		{"doc7", "pineapple mango orange"},
		{"doc8", "grape banana strawberry"},
		{"doc9", "blueberry raspberry ananas"},
		{"doc10", "orange mango banana orange orange orange ananas ananas"},
	}
}

func FindWordDoc(docs []Doc) map[string]int {
	var wordInDocs = map[string]int{}
	for _, doc := range docs {
		mp := make(map[string]int)
		docStrings := strings.Split(doc.Text, " ")
		for _, s := range docStrings {
			if _, ok := mp[s]; !ok {
				wordInDocs[s] += 1
			}
			mp[s]++
		}

	}
	return wordInDocs
}

func FindDocTF(docs []Doc, c chan []DocTF) {
	defer close(c)
	res := initTFSlice(len(docs))
	for i, doc := range docs {
		res[i].Name = doc.Name
		mp := make(map[string]int)
		docStrings := strings.Split(doc.Text, " ")
		for _, s := range docStrings {
			mp[s]++
		}
		for k, v := range mp {
			res[i].TF[k] = float64(v) / float64(len(docStrings))
		}
	}
	c <- res
}

func FindWordIdf(docs []Doc, c chan map[string]float64) map[string]float64 {
	defer close(c)
	wordInDocs := FindWordDoc(docs)
	res := make(map[string]float64)
	docsLen := len(docs)
	for k, v := range wordInDocs {
		res[k] = math.Log2(float64(docsLen) / float64(v))
	}
	c <- res
	return res
}

type TfIdf struct {
	Name  string
	TfIdf []map[string]float64
}

type SearchResult struct {
	DocName string
	Sum     float64
}

func search(searchInfo string, tfIdfs []TfIdf) []SearchResult {
	searchInf := strings.Split(searchInfo, " ")
	var res []SearchResult
	for _, idf := range tfIdfs {
		tempSum := 0.0
		for _, s := range searchInf {
			for _, mp := range idf.TfIdf {
				tempSum += mp[s]
			}
		}
		res = append(res, SearchResult{DocName: idf.Name, Sum: tempSum})
	}

	for i := 0; i < len(res); i++ {
		for j := 0; j < len(res)-1; j++ {
			if res[j].Sum < res[j+1].Sum {
				res[j], res[j+1] = res[j], res[j+1]
			}
		}
	}

	return res
}

func main() {
	tf := make(chan []DocTF)
	idf := make(chan map[string]float64)

	data := initTestDocs()
	go FindDocTF(data, tf)
	go FindWordIdf(data, idf)
	docsTf := <-tf
	wordsIdf := <-idf
	res := make([]TfIdf, len(docsTf))
	for i, docTF := range docsTf {
		res[i].Name = docTF.Name
		for key, val := range docTF.TF {
			res[i].TfIdf = append(res[i].TfIdf, map[string]float64{key: val * wordsIdf[key]})
		}
	}
	bytes, err := json.MarshalIndent(&res, "", " ")
	if err != nil {
		log.Println(err)
		return
	}
	os.WriteFile("model.json", bytes, 0644)

	results := search("apple orange", res)
	for _, result := range results {
		log.Printf("Doc: %s is get ok whith %.4f reality\n=========\n", result.DocName, result.Sum)
	}
	for _, tfIdf := range res {
		fmt.Printf("Doc: %s\nTF-IDF%+v\n===============\n", tfIdf.Name, tfIdf.TfIdf)
	}
}
