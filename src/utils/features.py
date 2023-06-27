import pandas as pd
from urllib.parse import urlparse, parse_qs
from tld import get_tld, get_fld
import requests
import dns.resolver
from ipwhois import IPWhois
from datetime import datetime
import ssl
from googlesearch import search
import re

class Features:

    """This class is used to extract the features from the URLs.

    Attributes:
        None    
    """

    def __init__(self):
        pass

    def get_qty_dot_url(
        self, 
        url: str
    ) -> int:
        
        return url.count('\.')

    def get_qty_hyphen_url(
        self, 
        url: str
    ) -> int:

        return url.count('-')

    def get_qty_underline_url(
        self, 
        url: str
    ) -> int:
        
        return url.count('_')
    
    def get_qty_slash_url(
        self, 
        url: str
    ) -> int:
        
        return url.count('/')
    
    def get_qty_questionmark_url(
        self, 
        url: str
    ) -> int:
        
        return url.count('\?')
    
    def get_qty_equal_url(
        self, 
        url: str
    ) -> int:
        
        return url.count('=')
    
    def get_qty_at_url(
        self, 
        url: str
    ) -> int:
        
        return url.count('@')

    def get_qty_and_url(
        self, 
        url: str
    ) -> int:
        
        return url.count('&')

    def get_qty_exclamation_url(
        self, 
        url: str
    ) -> int:
        
        return url.count('!')
    
    def get_qty_space_url(
        self, 
        url: str
    ) -> int:
        
        return url.count(' ')
    
    def get_qty_tilde_url(
        self, 
        url: str
    ) -> int:
        
        return url.count('~')

    def get_qty_comma_url(
        self, 
        url: str
    ) -> int:
        
        return url.count(',')

    def get_qty_plus_url(
        self, 
        url: str
    ) -> int:
        
        return url.count('\+')
    
    def get_qty_asterisk_url(
        self, 
        url: str
    ) -> int:
        
        return url.count('\*')
    
    def get_qty_hashtag_url(
        self, 
        url: str
    ) -> int:
        
        return url.count('#')
    
    def get_qty_dollar_url(
        self, 
        url: str
    ) -> int:
        
        return url.count('\$')
    
    def get_qty_percent_url(
        self, 
        url: str
    ) -> int:
        
        return url.count('%')
    
    def get_qty_tld_url(
        self, 
        url: str
    ) -> int:
        
        return len(urlparse(url).netloc.split('.'))

    def get_length_url(
        self, 
        url: str
    ) -> int:
        
        return len(url)

    def get_qty_dot_domain(
        self, 
        url: str
    ) -> int:
        
        parsed_url = urlparse(url)
        return parsed_url.netloc.count('.')

    def get_qty_hyphen_domain(
        self, 
        url: str
    ) -> int:
        
        parsed_url = urlparse(url)
        return parsed_url.netloc.count('-')

    def get_qty_underline_domain(
        self, 
        url: str
    ) -> int:
        
        parsed_url = urlparse(url)
        return parsed_url.netloc.count('_')

    def get_qty_slash_domain(
        self, 
        url: str
    ) -> int:
        
        parsed_url = urlparse(url)
        return parsed_url.netloc.count('/')
    
    def get_qty_questionmark_domain(
        self, 
        url: str
    ) -> int:
        
        parsed_url = urlparse(url)
        return parsed_url.netloc.count('\?')
    
    def get_qty_equal_domain(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        return parsed_url.netloc.count('=')
    
    def get_qty_at_domain(
        self, 
        url: str
    ) -> int:
        
        parsed_url = urlparse(url)
        return parsed_url.netloc.count('@')
    
    def get_qty_and_domain(
        self, 
        url: str
    ) -> int:
        
        parsed_url = urlparse(url)
        return parsed_url.netloc.count('&')
    
    def get_qty_exclamation_domain(
        self, 
        url: str
    ) -> int:
        
        parsed_url = urlparse(url)
        return parsed_url.netloc.count('!')
    
    def get_qty_space_domain(
        self, 
        url: str
    ) -> int:
        
        parsed_url = urlparse(url)
        return parsed_url.netloc.count(' ')
    
    def get_qty_tilde_domain(
        self, 
        url: str
    ) -> int:
        
        parsed_url = urlparse(url)
        return parsed_url.netloc.count('~')
    
    def get_qty_comma_domain(
        self, 
        url: str
    ) -> int:
        
        parsed_url = urlparse(url)
        return parsed_url.netloc.count(',')

    def get_qty_plus_domain(
        self, 
        url: str
    ) -> int:
        
        parsed_url = urlparse(url)
        return parsed_url.netloc.count('+')
    
    def get_qty_asterisk_domain(
        self, 
        url: str
    ) -> int:
        
        parsed_url = urlparse(url)
        return parsed_url.netloc.count('*')

    def get_qty_hashtag_domain(
        self, 
        url: str
    ) -> int:
        
        parsed_url = urlparse(url)
        return parsed_url.netloc.count('#')
    
    def get_qty_dollar_domain(
        self, 
        url: str
    ) -> int:
        
        parsed_url = urlparse(url)
        return parsed_url.netloc.count('$')
    
    def get_qty_percent_domain(
        self, 
        url: str
    ) -> int:
        
        parsed_url = urlparse(url)
        return parsed_url.netloc.count('%')
    
    def get_qty_vowels_domain(
        self, 
        url: str
    ) -> int:
        
        parsed_url = urlparse(url)
        parsed_domain = parsed_url.netloc.lower()  # Convert to lowercase for consistent counting
        return sum(parsed_domain.count(vowel) for vowel in 'aeiou')

    def get_domain_length(
        self, 
        url: str
    ) -> int:
        
        parsed_url = urlparse(url)
        return len(parsed_url.netloc)

    def get_domain_in_ip(
        self, 
        url: str    
    ) -> int:
        
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.split(':')[0]  # in case the URL contains a port number
        ip_pattern = r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
        if re.match(ip_pattern, domain):
            return 1
        else:
            return 0

    def get_server_client_domain(
        self, 
        url: str
    ) -> int:
        
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()  # Convert to lowercase for consistent checking
        if "server" in domain or "client" in domain:
            return 1
        else:
            return 0

    def get_qty_dot_directory(
        self, 
        url: str
    ) -> int:
    
        parsed_url = urlparse(url)
        return parsed_url.path.count('.')
    
    def get_qty_hyphen_directory(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        return parsed_url.path.count('-')

    def get_qty_underline_directory(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        return parsed_url.path.count('_')

    def get_qty_slash_directory(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        return parsed_url.path.count('/')
    
    def get_qty_questionmark_directory(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        return parsed_url.path.count('?')

    def get_qty_equal_directory(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        return parsed_url.path.count('=')
    
    def get_qty_at_directory(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        return parsed_url.path.count('@')

    def get_qty_and_directory(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        return parsed_url.path.count('&')

    def get_qty_exclamation_directory(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        return parsed_url.path.count('!')

    def get_qty_space_directory(
        self, 
        url: str
    ) -> int:
    
        parsed_url = urlparse(url)
        return parsed_url.path.count(' ')

    def get_qty_tilde_directory(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        return parsed_url.path.count('~')

    def get_qty_comma_directory(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        return parsed_url.path.count(',')

    def get_qty_plus_directory(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        return parsed_url.path.count('+')
    
    def get_qty_asterisk_directory(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        return parsed_url.path.count('*')
    
    def get_qty_hashtag_directory(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        return parsed_url.path.count('#')
    
    def get_qty_dollar_directory(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        return parsed_url.path.count('$')

    def get_qty_percent_directory(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        return parsed_url.path.count('%')

    def get_directory_length(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        return len(parsed_url.path)

    def get_qty_dot_file(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        # Splitting path by "/" to get the file (last part of the path).
        file = parsed_url.path.split('/')[-1]
        return file.count('.')

    def get_qty_hyphen_file(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        # Splitting path by "/" to get the file (last part of the path).
        file = parsed_url.path.split('/')[-1]
        return file.count('-')

    def get_qty_underline_file(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        # Splitting path by "/" to get the file (last part of the path).
        file = parsed_url.path.split('/')[-1]
        return file.count('_')

    def get_qty_slash_file(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        # Splitting path by "/" to get the file (last part of the path).
        file = parsed_url.path.split('/')[-1]
        return file.count('/')

    def get_qty_questionmark_file(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        # Splitting path by "/" to get the file (last part of the path).
        file = parsed_url.path.split('/')[-1]
        return file.count('?')

    def get_qty_equal_file(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        # Splitting path by "/" to get the file (last part of the path).
        file = parsed_url.path.split('/')[-1]
        return file.count('=')
        
    def get_qty_at_file(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        # Splitting path by "/" to get the file (last part of the path).
        file = parsed_url.path.split('/')[-1]
        return file.count('@')

    def get_qty_and_file(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        file = parsed_url.path.split('/')[-1]
        return file.count('&')

    def get_qty_exclamation_file(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        file = parsed_url.path.split('/')[-1]
        return file.count('!')

    def get_qty_space_file(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        file = parsed_url.path.split('/')[-1]
        return file.count(' ')
    
    def get_qty_tilde_file(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        file = parsed_url.path.split('/')[-1]
        return file.count('~')
    
    def get_qty_comma_file(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        file = parsed_url.path.split('/')[-1]
        return file.count(',')
    
    def get_qty_comma_file(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        file = parsed_url.path.split('/')[-1]
        return file.count(',')

    def get_qty_plus_file(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        file = parsed_url.path.split('/')[-1]
        return file.count('+')

    def get_qty_asterisk_file(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        file = parsed_url.path.split('/')[-1]
        return file.count('*')

    def get_qty_hashtag_file(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        file = parsed_url.path.split('/')[-1]
        return file.count('#')

    def get_qty_dollar_file(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        file = parsed_url.path.split('/')[-1]
        return file.count('$')

    def get_qty_percent_file(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        file = parsed_url.path.split('/')[-1]
        return file.count('%')

    def get_file_length(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        file = parsed_url.path.split('/')[-1]
        return len(file)

    def get_qty_dot_params(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        params = parsed_url.query
        return params.count('.')

    def get_qty_hyphen_params(
        self,
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        params = parsed_url.query
        return params.count('-')

    def get_qty_underline_params(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        params = parsed_url.query
        return params.count('_')

    def get_qty_slash_params(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        params = parsed_url.query
        return params.count('/')

    def get_qty_questionmark_params(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        params = parsed_url.query
        return params.count('?')

    def get_qty_equal_params(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        params = parsed_url.query
        return params.count('=')
    
    def get_qty_at_params(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        params = parsed_url.query
        return params.count('@')
    
    def get_qty_and_params(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        params = parsed_url.query
        return params.count('&')

    def get_qty_exclamation_params(
        self, 
        url: str
    ) -> int:
    
        parsed_url = urlparse(url)
        params = parsed_url.query
        return params.count('!')

    def get_qty_space_params(
        self, 
        url: str
    ) -> int:
    
        parsed_url = urlparse(url)
        params = parsed_url.query
        return params.count(' ')

    def get_qty_tilde_params(
        self, 
        url: str
    ) -> int:
    
        parsed_url = urlparse(url)
        params = parsed_url.query
        return params.count('~')

    def get_qty_comma_params(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        params = parsed_url.query
        return params.count(',')

    def get_qty_plus_params(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        params = parsed_url.query
        return params.count('+')

    def get_qty_asterisk_params(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        params = parsed_url.query
        return params.count('*')

    def get_qty_hashtag_params(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        params = parsed_url.query
        return params.count('#')
    
    def get_qty_dollar_params(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        params = parsed_url.query
        return params.count('$')

    def get_qty_percent_params(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        params = parsed_url.query
        return params.count('%')
    
    def get_params_length(self, url: str) -> int:
        parsed_url = urlparse(url)
        return len(parsed_url.query)

    def get_tld_present_params(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        params = parsed_url.query

        # Split parameters by '&' and '='
        parts = params.replace('&', ' ').replace('=', ' ').split()

        for part in parts:
            try:
                get_fld(part, fail_silently=False)
                return 1
            except:
                pass
        return 0

    def get_qty_params(
        self, 
        url: str
    ) -> int:

        parsed_url = urlparse(url)
        params = parse_qs(parsed_url.query)
        return len(params)
    
    def get_email_in_url(
        self, 
        url: str
    ) -> int:

        """get_email_in_url Checks if there is an email address in the URL

        Args:
            url (str): URL to be analyzed

        Returns:
            int: 1 if there is an email address in the URL, 0 otherwise
        """

        parsed_url = urlparse(url)
        params = parsed_url.query

        # Split parameters by '&' and '='
        parts = params.replace('&', ' ').replace('=', ' ').split()

        # Regular expression pattern for an email address
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

        for part in parts:
            if re.fullmatch(pattern, part):
                return 1
        return 0

    def get_time_response(
        self, 
        url: str
    ) -> int:

        """get_time_response Gets the response time of the URL

        Args:
            url (str): URL to be analyzed

        Returns:
            int: Response time in seconds
        """

        try:
            if not url.startswith('http'):
                url = 'http://' + url  # Prepend 'http://' if missing
            response = requests.get(url)
            return response.elapsed.total_seconds()
        except requests.exceptions.RequestException:
            return 0
    
    def get_domain_spf(
        self, 
        url: str
    ) -> int:

        """get_domain_spf Gets the SPF record of the domain

        Args:
            url (str): URL to be analyzed

        Returns:
            int: 1 if SPF record exists, 0 if not, -1 if error
        """

        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc:
                domain = parsed_url.netloc
            else:
                domain = parsed_url.path.split('/')[0]
            answers = dns.resolver.resolve(domain, 'TXT')
            for rdata in answers:
                txt_data = rdata.to_text()
                if 'v=spf1' in txt_data:
                    return 1
            return 0
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers):
            return -1

    def get_asn_ip(
        self, 
        url: str
    ) -> int:

        """get_asn_ip Gets the ASN of the IP address

        Args:
            url (str): URL to be analyzed
        
        Returns:
            int: ASN of the IP address
        """

        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc:
                domain = parsed_url.netloc
            else:
                domain = parsed_url.path.split('/')[0]
            answers = dns.resolver.resolve(domain, 'A')
            for rdata in answers:
                ip = rdata.to_text()
                obj = IPWhois(ip)
                result = obj.lookup_whois()
                asn = result.get('asn')
                if asn:
                    return asn
            return 0
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers):
            return -1

    def get_time_domain_activation(
        self, 
        url: str
    ) -> int:

        """get_time_domain_activation Gets the time of domain activation

        Args:
            url (str): URL to be analyzed
        
        Returns:
            int: Time of domain activation in seconds
        """

        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc:
                domain = parsed_url.netloc
            else:
                domain = parsed_url.path.split('/')[0]
            answers = dns.resolver.resolve(domain, 'A')
            for rdata in answers:
                ip = rdata.to_text()
                obj = IPWhois(ip)
                result = obj.lookup_whois()
                asn_date = result.get('asn_date')
                if asn_date:
                    asn_datetime = datetime.strptime(asn_date, "%Y-%m-%d")
                    return int(asn_datetime.timestamp())
            return 0
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers):
            return -1

    def get_time_domain_expiration(
        self, 
        url: str
    ) -> int:

        """get_time_domain_expiration Gets the time of domain expiration

        Args:
            url (str): URL to be analyzed
        
        Returns:
            int: Time of domain expiration
        
        """

        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc:
                domain = parsed_url.netloc
            else:
                domain = parsed_url.path.split('/')[0]
            answers = dns.resolver.resolve(domain, 'A')
            for rdata in answers:
                ip = rdata.to_text()
                obj = IPWhois(ip)
                result = obj.lookup_whois()
                asn_date = result.get('asn_date')
                if asn_date:
                    asn_datetime = datetime.strptime(asn_date, "%Y-%m-%d")
                    return int(asn_datetime.timestamp())
            return 0
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers):
            return -1

    def get_qty_ip_resolved(
        self, 
        url: str
    ) -> int:

        """get_qty_ip_resolved Gets the quantity of IP resolved for a URL

        Args:
            url (str): URL to get the quantity of IP resolved

        Returns:
            int: Quantity of IP resolved
        """

        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc:
                domain = parsed_url.netloc
            else:
                domain = parsed_url.path.split('/')[0]
            answers = dns.resolver.resolve(domain, 'A')
            return len(answers)
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers):
            return 0

    def get_qty_nameservers(
        self,
        url: str
    ) -> int:

        """get_qty_nameservers Gets the quantity of nameservers for a URL

        Args:   
            url (str): URL to get the quantity of nameservers

        Returns:    
            int: Quantity of nameservers
        """

        try:
            parsed_url = urlparse(url)
            if not parsed_url.netloc:
                domain = parsed_url.path.split('/')[0]
            else:
                domain = parsed_url.netloc
            answers = dns.resolver.resolve(domain, 'NS')
            return len(answers)
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers):
            return -1

    def get_qty_mx_servers(
        self, 
        url: str
    ) -> int:

        """get_qty_mx_servers Gets the quantity of MX servers for a URL

        Args:
            url (str): URL to get the quantity of MX servers

        Returns:
            int: Quantity of MX servers
        """

        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc:
                domain = parsed_url.netloc
            else:
                domain = parsed_url.path.split('/')[0]
            answers = dns.resolver.resolve(domain, 'MX')
            return len(answers)
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers):
            return -1

    def get_ttl_hostname(
        self, 
        url: str
    ) -> int:

        """get_ttl_hostname Gets the TTL of the hostname for a URL

        Args:
            url (str): URL to get the TTL of the hostname

        Returns:
            int: TTL of the hostname
        """

        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc:
                domain = parsed_url.netloc
            else:
                domain = parsed_url.path.split('/')[0]
            answers = dns.resolver.resolve(domain, 'A')
            for rdata in answers:
                ip = rdata.to_text()
                obj = IPWhois(ip)
                result = obj.lookup_whois()
                asn_date = result.get('asn_date')
                if asn_date:
                    asn_datetime = datetime.strptime(asn_date, "%Y-%m-%d")
                    return int(asn_datetime.timestamp())
            return 0
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.NoNameservers):
            return -1

    def get_tls_ssl_certificate(
        self, 
        url: str
    ) -> int:

        """get_tls_ssl_certificate Gets the TLS/SSL certificate for a URL

        Args:
            url (str): URL to check
        
        Returns:    
            int: 1 if the certificate exists and is not expired, 0 if the certificate exists but is expired, -1 if the certificate does not exist
        """

        try:
            parsed_url = urlparse(url)
            hostname = parsed_url.netloc
            
            # Try HTTPS connection
            context = ssl.create_default_context()
            with context.wrap_socket(socket.socket(), server_hostname=hostname) as sock:
                try:
                    sock.connect((hostname, 443))
                    cert = sock.getpeercert()
                    
                    # Check if the certificate exists and is not expired
                    if cert and 'notAfter' in cert:
                        return 1
                except:
                    pass
            
            # Try HTTP connection
            with socket.create_connection((hostname, 80)) as sock:
                return 1
            
            return 0
        except:
            return -1

    def get_qty_redirects(
        self, 
        url: str
    ) -> int:

        """get_qty_redirects Gets the number of redirects for a URL

        Args:
            url (str): URL to check

        Returns:
            int: Number of redirects, 0 if no redirects, -1 if an error occurred

        """

        try:
            response = requests.get(url, allow_redirects=True)
            return len(response.history)
        except requests.exceptions.RequestException:
            return 0

    def get_url_google_index(
        self,
        url: str
    ) -> int:

        """get_url_google_index Checks if the URL is indexed by Google

        Args:
            url (str): URL to check 
        
        Returns:
            int: 1 if the URL is indexed, 0 if not, -1 if an error occurred

        """

        try:
            query = f'site:{url}'
            results = search(query, num_results=10)  # Adjust the number of results as needed
            for i, result in enumerate(results, start=1):
                if url == result:
                    return i
            return 0
        except:
            return -1

    def get_domain_google_index(
        self, 
        url: str
    ) -> int:

        """get_domain_google_index Checks if the domain is indexed by Google

        Args:
            url (str): URL to check

        Returns:
            int: 1 if the domain is indexed, 0 if not, -1 if an error occurred

        """

        try:
            domain = url.split('//')[-1].split('/')[0]
            query = f'site:{domain}'
            results = search(query, num_results=10)  # Adjust the number of results as needed
            for i, result in enumerate(results, start=1):
                if domain in result:
                    return i
            return 0
        except:
            return -1

    def get_url_shortened(
        self, 
        url: str
    ) -> int:

        """get_url_shortened Checks if the URL is shortened

        Args:
            url (str): URL to check
        
        Returns:
            int: 1 if the URL is shortened, 0 if not, -1 if an error occurred
        
        """

        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc:
                domain = parsed_url.netloc
            else:
                domain = parsed_url.path.split('/')[0]
            if domain in self.url_shortened:
                return 1
            return 0
        except:
            return -1

    def get_url_shortened(
        self, 
        url: str
    ) -> int:

        """get_url_shortened Checks if the URL is shortened

        Args:
            url (str): URL to check
        
        Returns:
            int: 1 if the URL is shortened, 0 if not, -1 if an error occurred
        
        """

        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme:
                url = "https://" + url
            response = requests.head(url, allow_redirects=True)
            resolved_url = response.url
            if len(url) > len(resolved_url):
                return 1
            return 0
        except requests.exceptions.RequestException:
            return -1

    def extract_features(
        self, 
        url: str, 
        source_data: str, 
        top_features: int = 40
    ) -> dict:

        """extract_features Extracts features from a URL

        Args:
            url (str): URL to extract features from
            source_data (str): Path to the source data
            top_features (int, optional): Number of top features to extract. Defaults to 40.
        
        Returns:
            dict: Dictionary of features

        """

        df = pd.read_csv(source_data)

        columns = df.corr()['phishing'].sort_values(ascending=False)[:top_features].index.tolist()

        feature_functions = [getattr(self, func) for func in dir(self) if func.startswith('get') and callable(getattr(self, func)) and func.split('get_')[-1] in columns]

        feature_results = {}

        for func in feature_functions:
            result = func(url=url)
            feature_name = func.__name__
            feature_results[feature_name] = result

        sorted_feature_results = dict(sorted(feature_results.items(), key=lambda x: columns.index(x[0].split('get_')[-1])))

        sorted_feature_results = {key.split("get_", 1)[1]: value for key, value in sorted_feature_results.items()}

        return sorted_feature_results