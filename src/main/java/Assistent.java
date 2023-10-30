import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;
import dev.langchain4j.retriever.EmbeddingStoreRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.TokenStream;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Paths;
import java.time.Duration;

import static dev.langchain4j.data.document.FileSystemDocumentLoader.loadDocument;

interface SustainableArchitectureAssistent {
    TokenStream beantworte(String frage);
}

public class Assistent {

    public static void main(String[] args) throws Exception {
        var embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        var ingestor = EmbeddingStoreIngestor
                .builder()
                .documentSplitter(DocumentSplitters.recursive(500, 0))
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build();

        ingestor.ingest(loadDocument(
                Paths.get(Assistent.class.getResource("Sustainable_Architecture_Guidelines.pdf").toURI())
        ));

        var gpt = OpenAiStreamingChatModel
                .builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .timeout(Duration.ofMinutes(1))
                .build();

        var assistant = AiServices
                .builder(SustainableArchitectureAssistent.class)
                .streamingChatLanguageModel(gpt)
                .retriever(EmbeddingStoreRetriever.from(embeddingStore, embeddingModel))
                .build();

        System.out.println("Bin bereit!");

        while (true) {
            var tokenStream = assistant.beantworte(
                    new BufferedReader(new InputStreamReader(System.in)).readLine()
            );

            tokenStream
                    .onNext(System.out::print)
                    .onComplete((in) -> System.out.println("\n==========================================="))
                    .onError(Throwable::printStackTrace)
                    .start();
        }
    }
}
