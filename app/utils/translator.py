from typing import Optional
import requests

class Translator:
    """Translates text from any language to English."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the translator.
        
        Args:
            api_key: Optional API key for translation service
        """
        self.api_key = api_key
    
    def translate(self, text: str, source_language: str = "auto") -> str:
        """
        Translate text to English.
        
        Args:
            text: The text to translate
            source_language: Source language code (default: "auto" for auto-detection)
        
        Returns:
            Translated text in English
        """
        try:
            url = "https://api.mymemory.translated.net/get"
            params = {
                "q": text,
                "langpair": f"{source_language}|en"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data["responseStatus"] == 200:
                return data["responseData"]["translatedText"]
            else:
                return f"Translation error: {data.get('responseDetails', 'Unknown error')}"
        
        except requests.RequestException as e:
            return f"Error during translation: {str(e)}"


# Example usage
if __name__ == "__main__":
    translator = Translator()
    result = translator.translate("Hola, ¿cómo estás?")
    print(result)